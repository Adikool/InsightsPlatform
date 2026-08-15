"""Minimal Apache Superset REST client (API v1, Superset 3.x / 4.x).

Auth is a two-step dance that trips people up: a JWT from `/security/login` is not
enough — mutating calls also need a CSRF token *and* the session cookie that was
issued alongside it, plus a Referer header. Hence the shared `requests.Session`.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from ..config import settings
from ..errors import SupersetError


class SupersetClient:
    def __init__(
        self,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        provider: str | None = None,
        timeout: int = 30,
        verify: bool = True,
    ) -> None:
        self.base_url = (base_url or settings.superset_url).rstrip("/")
        self.username = username or settings.superset_username
        self.password = password or settings.superset_password
        self.provider = provider or settings.superset_provider
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = verify
        self._token: str | None = None
        self._csrf: str | None = None

    # ------------------------------------------------------------------ auth
    def login(self) -> None:
        url = f"{self.base_url}/api/v1/security/login"
        try:
            response = self.session.post(
                url,
                json={
                    "username": self.username,
                    "password": self.password,
                    "provider": self.provider,
                    "refresh": True,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise SupersetError(f"cannot reach Superset at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise SupersetError(
                f"Superset login failed ({response.status_code}): {response.text[:300]}"
            )
        self._token = response.json().get("access_token")
        if not self._token:
            raise SupersetError("Superset login returned no access_token")

        csrf = self.session.get(
            f"{self.base_url}/api/v1/security/csrf_token/",
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=self.timeout,
        )
        if csrf.status_code == 200:
            self._csrf = csrf.json().get("result")

    def _headers(self, mutating: bool) -> dict[str, str]:
        if not self._token:
            self.login()
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        if mutating and self._csrf:
            headers["X-CSRFToken"] = self._csrf
            headers["Referer"] = self.base_url
        return headers

    # --------------------------------------------------------------- plumbing
    def _request(self, method: str, path: str, **kwargs) -> Any:
        mutating = method.upper() in ("POST", "PUT", "DELETE")
        url = f"{self.base_url}/api/v1{path}"
        try:
            response = self.session.request(
                method, url, headers=self._headers(mutating), timeout=self.timeout, **kwargs
            )
        except requests.RequestException as exc:
            raise SupersetError(f"{method} {path} failed: {exc}") from exc

        if response.status_code == 401 and self._token:
            # Token expired mid-run; one silent re-login is worth it.
            self._token = None
            response = self.session.request(
                method, url, headers=self._headers(mutating), timeout=self.timeout, **kwargs
            )

        if response.status_code >= 400:
            raise SupersetError(
                f"{method} {path} -> {response.status_code}: {response.text[:500]}"
            )
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {"raw": response.text}

    def get(self, path: str, **kwargs) -> Any:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, payload: dict) -> Any:
        return self._request("POST", path, json=payload)

    def put(self, path: str, payload: dict) -> Any:
        return self._request("PUT", path, json=payload)

    # ------------------------------------------------------------------- ping
    def ping(self) -> dict:
        self.login()
        me = self.get("/me/")
        return me.get("result", me)

    # -------------------------------------------------------------- databases
    def find_database(self, name: str) -> dict | None:
        query = '{"filters":[{"col":"database_name","opr":"eq","value":%s}]}' % _json_str(name)
        result = self.get("/database/", params={"q": query})
        items = result.get("result") or []
        return items[0] if items else None

    def ensure_database(self, name: str, sqlalchemy_uri: str) -> int:
        existing = self.find_database(name)
        if existing:
            return int(existing["id"])
        created = self.post(
            "/database/",
            {
                "database_name": name,
                "sqlalchemy_uri": sqlalchemy_uri,
                "expose_in_sqllab": True,
                "allow_ctas": False,
                "allow_cvas": False,
                "allow_dml": False,
                "allow_run_async": False,
            },
        )
        return int(created["id"])

    # --------------------------------------------------------------- datasets
    def find_dataset(self, table_name: str, database_id: int | None = None) -> dict | None:
        filters = [{"col": "table_name", "opr": "eq", "value": table_name}]
        if database_id is not None:
            filters.append({"col": "database", "opr": "rel_o_m", "value": database_id})
        result = self.get("/dataset/", params={"q": _rison(filters)})
        items = result.get("result") or []
        return items[0] if items else None

    def create_dataset(
        self,
        database_id: int,
        table_name: str,
        sql: str | None = None,
        schema: str | None = None,
    ) -> int:
        payload: dict[str, Any] = {
            "database": database_id,
            "table_name": table_name,
            "owners": [],
        }
        if schema:
            payload["schema"] = schema
        if sql:
            payload["sql"] = sql  # virtual dataset
        created = self.post("/dataset/", payload)
        return int(created["id"])

    def refresh_dataset(self, dataset_id: int) -> None:
        """Force Superset to re-read columns/metrics for a virtual dataset."""
        try:
            self.put(f"/dataset/{dataset_id}/refresh", {})
        except SupersetError:
            # Not fatal: the dataset still works, the column list may just lag.
            pass

    def dataset_columns(self, dataset_id: int) -> list[dict]:
        result = self.get(f"/dataset/{dataset_id}")
        return result.get("result", {}).get("columns", [])

    # ----------------------------------------------------------------- charts
    def create_chart(
        self,
        name: str,
        viz_type: str,
        dataset_id: int,
        params: dict,
        description: str = "",
    ) -> int:
        import json

        created = self.post(
            "/chart/",
            {
                "slice_name": name,
                "viz_type": viz_type,
                "datasource_id": dataset_id,
                "datasource_type": "table",
                "params": json.dumps(params),
                "description": description,
                "owners": [],
            },
        )
        return int(created["id"])

    def attach_chart(self, chart_id: int, dashboard_id: int) -> None:
        current = self.get(f"/chart/{chart_id}").get("result", {})
        dashboards = [d["id"] for d in current.get("dashboards", [])]
        if dashboard_id not in dashboards:
            dashboards.append(dashboard_id)
        self.put(f"/chart/{chart_id}", {"dashboards": dashboards})

    # ------------------------------------------------------------- dashboards
    def find_dashboard(self, title: str) -> dict | None:
        filters = [{"col": "dashboard_title", "opr": "eq", "value": title}]
        result = self.get("/dashboard/", params={"q": _rison(filters)})
        items = result.get("result") or []
        return items[0] if items else None

    def ensure_dashboard(self, title: str) -> int:
        existing = self.find_dashboard(title)
        if existing:
            return int(existing["id"])
        return self.create_dashboard(title)

    def create_dashboard(self, title: str) -> int:
        created = self.post(
            "/dashboard/", {"dashboard_title": title, "published": True, "owners": []}
        )
        return int(created["id"])

    def dashboard_is_occupied(self, dashboard_id: int) -> bool:
        """Does this dashboard already have a layout someone else built?

        Publishing replaces `position_json` wholesale, which silently orphans every
        chart already arranged on it. Reusing a dashboard we created is fine;
        reusing one that already had a layout destroys work, so callers must ask.
        """
        try:
            result = self.get(f"/dashboard/{dashboard_id}").get("result", {})
        except SupersetError:
            return False
        raw = result.get("position_json") or ""
        if not raw.strip():
            return False
        try:
            import json

            position = json.loads(raw)
        except ValueError:
            return False
        return any(key.startswith("CHART-") for key in position)

    def unique_dashboard_title(self, title: str, limit: int = 50) -> str:
        """`Sales Dashboard` -> `Sales Dashboard (2)` when the name is taken."""
        if not self.find_dashboard(title):
            return title
        for suffix in range(2, limit):
            candidate = f"{title} ({suffix})"
            if not self.find_dashboard(candidate):
                return candidate
        raise SupersetError(f"could not find a free dashboard title based on {title!r}")

    def detach_chart(self, chart_id: int, dashboard_id: int) -> None:
        current = self.get(f"/chart/{chart_id}").get("result", {})
        remaining = [d["id"] for d in current.get("dashboards", []) if d["id"] != dashboard_id]
        self.put(f"/chart/{chart_id}", {"dashboards": remaining})

    # ------------------------------------------------------------------- urls
    def chart_url(self, chart_id: int) -> str:
        return f"{self.base_url}/explore/?slice_id={chart_id}"

    def dashboard_url(self, dashboard_id: int) -> str:
        return f"{self.base_url}/superset/dashboard/{dashboard_id}/"


def _json_str(value: str) -> str:
    import json

    return json.dumps(value)


def _rison(filters: list[dict]) -> str:
    """Superset list endpoints take a Rison query. JSON is accepted for these
    simple shapes, which avoids a Rison dependency."""
    import json

    return json.dumps({"filters": filters})
