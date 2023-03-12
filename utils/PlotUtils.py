import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
    # Display the figure
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
