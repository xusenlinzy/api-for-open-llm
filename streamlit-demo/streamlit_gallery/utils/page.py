from typing import Callable, Optional

import streamlit as st


def page_group(param):
    key = f"{__name__}_page_group_{param}"

    if key not in st.session_state:
        st.session_state.update({key: PageGroup(param)})

    return st.session_state[key]


class PageGroup:

    def __init__(self, param):
        self._param: str = param
        self._default = None
        self._selected = None

        # Fix some rollback issues when multiple pages are selected in the same run.
        self._backup: Optional[str] = None

    @property
    def selected(self):
        params = st.query_params.to_dict()
        return params[self._param] if self._param in params else self._default

    def item(self, label: str, callback: Callable, default=False) -> None:
        self._backup = None

        key = f"{__name__}_{self._param}_{label}"
        page = self._normalize_label(label)

        if default:
            self._default = page

        selected = (page == self.selected)

        if selected:
            self._selected = callback

        st.session_state[key] = selected
        st.checkbox(label, key=key, disabled=selected, on_change=self._on_change, args=(page,))

    def show(self) -> None:
        if self._selected is not None:
            self._selected()
        else:
            st.title("🤷 404 Not Found")

    def _on_change(self, page: str) -> None:
        params = st.query_params.to_dict()

        if self._backup is None:
            if self._param in params:
                self._backup = params[self._param][0]
            params[self._param] = [page]
        else:
            params[self._param] = [self._backup]

        for key in params:
            st.query_params[key] = params[key]
        st.session_state.messages = []

    def _normalize_label(self, label: str) -> str:
        return "".join(char.lower() for char in label if char.isascii()).strip().replace(" ", "-")
