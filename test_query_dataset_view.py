import requests, json, time
import pandas as pd
import numpy as np
from progress.bar import *
import matplotlib.pyplot as plt
from nicegui import ui

def df_preproc(dfm):
    #dfm = dfm.drop(["title", ], axis=1)
    dfm = dfm.loc[:, ~dfm.columns.str.contains('^Unnamed')]
    dfm = dfm.dropna(subset=['text'])
    dfm = dfm[dfm["text"].str.strip() != ""]
    return dfm

# Stat values
"""r = requests.post("http://127.0.0.1:5000/classify", json={"text": row["text"]})
results = r.json()
"""

#df_test = df_preproc(pd.read_csv("data/test.csv"))
df_test = pd.read_csv("data/orig/WELFake_Dataset.csv")
sample_articles = df_test.sample(n=20)

editor_val = ""
editor_results = {}
editor_dialog = None

@ui.refreshable
def main_view():
    global editor_val, editor_results

    @ui.refreshable
    def wc_cc_widget():
        try:
            ui.markdown("Character count: {}, Word count: {}".format(len(editor_val), len(editor_val.split(' '))))
        except TypeError as e:
            print("Type value error")

    with ui.column().classes('w-full px-5'):
        ui.markdown("### Dataset View")
        ui.markdown("Click on a table row to open it's article for **editing and processing**.")

        with ui.element('div').classes('columns-2 w-full gap-2'):
            with ui.element('div').classes('mb-2 px-2 rounded'):
                def update_wc(e):
                    global editor_val
                    editor_val = e.value
                    wc_cc_widget.refresh()

                editor = ui.editor(placeholder='Type something here...', on_change=update_wc)

            with ui.element('div').classes('mb-2 px-2 h-72 break-inside-avoid rounded'):
                table = ui.aggrid({
                    'columnDefs': [{'field': col} for col in sample_articles.columns],
                    'rowData': sample_articles.to_dict('records'),
                })

                with ui.row().classes('pt-5'):
                    ui.space()
                    wc_cc_widget()

                    @ui.refreshable
                    def refreshable_json():
                        global editor_dialog, editor_results
                        with ui.dialog() as editor_dialog, ui.card():
                            ui.markdown("#### Results")
                            ui.json_editor({'content': {'json': editor_results}})
                            ui.button('Close', on_click=editor_dialog.close)

                    refreshable_json()

                    def process_query():
                        global editor_val, editor_results

                        r = requests.post("http://127.0.0.1:8003/classify", json={"text": editor_val})
                        editor_results = r.json()
                        refreshable_json.refresh()

                        editor_dialog.open()

                    ui.button("Send to Server", on_click=process_query)

            def handle_click(sender):
                editor.value = sender.args["value"]

            table.on('cellClicked', handle_click)

# App Run
main_view()

ui.run()
ui.dark_mode().enable()