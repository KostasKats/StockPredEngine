import os
from datetime import datetime

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


def plotCurrentStatus(last_30_days, id):
    # Create a Plotly figure object
    fig = go.Figure()

    # Add a line trace for the adjusted close prices
    fig.add_trace(go.Scatter(x=last_30_days.index, y=last_30_days["Adj Close"], name="Adjusted Close"))

    # Set the layout of the figure
    fig.update_layout(
        title=f"{id} Stock Data for Last 30 Days",
        xaxis_title="Date",
        yaxis_title="Adjusted Close Price",
        hovermode="x"
    )
    fig.show()
def plotCurrentModelResults(y_test_scaled,predictions,id, column):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_facecolor('#000041')
    ax.plot(y_test_scaled, color='red', label='Original price')
    ax.plot(predictions, color='cyan', label='Predicted price')
    ax.set_xlabel('Time')
    ax.set_ylabel(f"Price: {column}" )
    ax.legend()
    plt.title(str(id).capitalize())
    plt.show()

def plotFutureSteps(y_test_scaled, predictions,future_predictions,id,column):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_facecolor('#000041')
    ax.plot(y_test_scaled, color='red', label='Original price')
    ax.plot(predictions, color='cyan', label='Predicted price')
    ax.plot(range(len(y_test_scaled), len(y_test_scaled) + len(future_predictions)), future_predictions, color='green',
            label='Future predicted price')
    ax.set_xlabel('Time')
    ax.set_ylabel(f"Price: {column}" )
    ax.legend()
    plt.title(str(id).capitalize())
    plt.show()



def plotFuture(y_test_scaled, predictions, future_predictions, id, column):
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Scatter(x=list(range(len(y_test_scaled))), y=y_test_scaled.reshape(-1), mode='lines', name='Original price',
                   line=dict(color='red')))
    fig.add_trace(
        go.Scatter(x=list(range(len(predictions))), y=predictions.reshape(-1), mode='lines', name='Predicted price',
                   line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=list(range(len(y_test_scaled), len(y_test_scaled) + len(future_predictions))),
                             y=future_predictions.reshape(-1), mode='lines', name='Future predicted price',
                             line=dict(color='green')))

    fig.update_layout(title=str(id).capitalize(), xaxis_title='Time', yaxis_title=f"Price: {column}",
                      plot_bgcolor='#000041', legend=dict(x=0.05, y=0.95))

    fig.show()

    print("Future Predictions:")
    print(future_predictions)

def savePlotFuture(y_test_scaled, predictions, future_predictions, id, column):
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Scatter(x=list(range(len(y_test_scaled))), y=y_test_scaled.reshape(-1), mode='lines',
                   name='Original price',
                   line=dict(color='red')))
    fig.add_trace(
        go.Scatter(x=list(range(len(predictions))), y=predictions.reshape(-1), mode='lines', name='Predicted price',
                   line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=list(range(len(y_test_scaled), len(y_test_scaled) + len(future_predictions))),
                             y=future_predictions.reshape(-1), mode='lines', name='Future predicted price',
                             line=dict(color='green')))

    fig.update_layout(title=str(id).capitalize(), xaxis_title='Time (days)', yaxis_title=f"Price: {column}",
                      plot_bgcolor='#000041', legend=dict(x=0.05, y=0.95))

    savePlot(id,fig)


def savePlot(id,fig):
    folder_path = f"../resources/{id}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig.write_html(os.path.join(folder_path, f"{datetime.today().strftime('%Y-%m-%d')}_{id}.html"))




