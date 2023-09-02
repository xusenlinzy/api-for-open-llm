from pathlib import Path

import streamlit as st


def main():
    with open(Path(__file__).parents[3]/"docs/DISPLAY.md", "r", encoding="utf8") as f:
        text = f.read()
    st.markdown(text)


if __name__ == "__main__":
    main()
