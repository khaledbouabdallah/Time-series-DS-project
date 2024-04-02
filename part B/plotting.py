import numpy as np

import plotly.graph_objects as go

# function that creates a scatter plot
def create_figure():
    # Generate dataset
    np.random.seed(1)
    x0 = np.random.normal(2, 0.4, 400)
    y0 = np.random.normal(2, 0.4, 400)
    x1 = np.random.normal(3, 0.6, 600)
    y1 = np.random.normal(6, 0.4, 400)
    x2 = np.random.normal(4, 0.2, 200)
    y2 = np.random.normal(4, 0.4, 200)

    # Create figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=x0,
            y=y0,
            mode="markers",
            marker=dict(color="DarkOrange")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y1,
            mode="markers",
            marker=dict(color="Crimson")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x2,
            y=y2,
            mode="markers",
            marker=dict(color="RebeccaPurple")
        )
    )

    # Add buttons that add shapes
    cluster0 = [dict(type="circle",
                                xref="x", yref="y",
                                x0=min(x0), y0=min(y0),
                                x1=max(x0), y1=max(y0),
                                line=dict(color="DarkOrange"))]
    cluster1 = [dict(type="circle",
                                xref="x", yref="y",
                                x0=min(x1), y0=min(y1),
                                x1=max(x1), y1=max(y1),
                                line=dict(color="Crimson"))]
    cluster2 = [dict(type="circle",
                                xref="x", yref="y",
                                x0=min(x2), y0=min(y2),
                                x1=max(x2), y1=max(y2),
                                line=dict(color="RebeccaPurple"))]

    fig.update_layout(
        updatemenus=[
            dict(buttons=list([
                dict(label="None",
                     method="relayout",
                     args=["shapes", []]),
                dict(label="Cluster 0",
                     method="relayout",
                     args=["shapes", cluster0]),
                dict(label="Cluster 1",
                     method="relayout",
                     args=["shapes", cluster1]),
                dict(label="Cluster 2",
                     method="relayout",
                     args=["shapes", cluster2]),
                dict(label="All",
                     method="relayout",
                     args=["shapes", cluster0 + cluster1 + cluster2])
            ]))
        ]
    )

    # Update remaining layout properties
    fig.update_layout(
        title_text="Highlight Clusters",
        showlegend=False,
    )

    return fig



# function that creates a line plot with interactive column choice using two datasets
def create_figure_two(df1,df2,):
    # Create figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(x=df1.index, y=df1["6"], mode="lines", name="Original")
    )
    fig.add_trace(
        go.Scatter(x=df2.index, y=df2["6"], mode="lines", name="Imputed")
    )

    # Add buttons that add shapes
    buttons = []

    for column in df1.columns:
        buttons.append(
            dict(
                label=column,
                method="restyle",
                args=["y", [df1[column],df2[column]]],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.3,
                yanchor="top"
            )
        ]
    )

    return fig

    