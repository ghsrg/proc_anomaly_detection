import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from src.utils.graph_utils import clean_graph


def visualize_graph_with_plotly(graph, proc_id='NA', file_path=None):
    """
    Візуалізація графа NetworkX у стилі BPMN із використанням Plotly.
    У вузлах відображаємо BPMN name або task_subject, фарбуємо вузли за типом.
    На ребрах показуємо conditionExpression або name (якщо є).
    Якщо SEQUENCE_COUNTER_ заповнено, вузол матиме чорну обводку.
    :param graph: Граф NetworkX для візуалізації.
    :param file_path: Шлях до файлу для збереження (якщо None, виводить на екран).
    """
   # graph = clean_graph(graph)
    pos = nx.spring_layout(graph)
    node_labels = {}
    node_colors = []
    border_colors = []
    node_hover_texts = []

    color_map = {
        'starteventsp': 'lightgreen',
        'startevent': 'green',
        'endeventsp': '#f9cfcf',
        'endevent': 'red',
        'gateway': 'yellow',
        'subprocess': 'cornflowerblue',
        'callactivity': 'cornflowerblue',
        'usertask': 'blue',
        'scripttask': '#f2e6fb',
        'servicetask': '#f2e6fb',
        'intermediatethrowevent': '#27c4c4',
        'boundaryevent': '#e7d0f7'
    }

    for node, data in graph.nodes(data=True):
        bpmn_name = data.get('name', node)
        node_type = data.get('type', '').lower() if isinstance(data.get('type', ''), str) else ''
        node_labels[node] = bpmn_name

        node_colors.append(color_map.get(node_type, 'lightblue'))
        seq_counter = data.get('SEQUENCE_COUNTER_')
        border_colors.append('black' if seq_counter else None)

        hover_text = f"Node: {node}<br>"
        hover_text += '<br>'.join([f"{key}: {value}" for key, value in data.items()])
        node_hover_texts.append(hover_text)

    edge_traces = []
    for u, v, edge_data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        cond_expr = edge_data.get('DURATION_', '')
        taskaction = edge_data.get('taskaction_code', '')
        flow_name = edge_data.get('name', '')
        edge_label = f'{cond_expr} \n {taskaction} \n {flow_name}'

        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines+markers',
            line=dict(width=1, color='gray'),
            marker=dict(symbol='arrow', size=10, color='black'),
            hoverinfo='text',
            text=edge_label
        ))

    fig = go.Figure()

    for trace in edge_traces:
        fig.add_trace(trace)

    for idx, (node, (x, y)) in enumerate(pos.items()):
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(
                size=15,
                color=node_colors[idx],
                line=dict(width=2, color=border_colors[idx]) if border_colors[idx] else dict(width=0)
            ),
            text=node_labels[node],
            textposition='top center',
            hoverinfo='text+name',
            name=node_labels[node],
            customdata=[node_hover_texts[idx]],
            hovertemplate='%{customdata}'
        ))

    fig.update_layout(
        title=f'BPMN Graph {proc_id}',
        showlegend=False,
        plot_bgcolor='white',  # Колір фону графіка
        paper_bgcolor='white',  # Колір фону всієї фігури
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        hovermode='closest'
    )

    if file_path:
        fig.write_html(file_path)
        print(f"Граф збережено у {file_path}")
    else:
        fig.show()
