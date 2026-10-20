import heapq
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog


st.set_page_config(page_title="Aiding Allies | Network Optimization", page_icon="✈️", layout="wide")


# -----------------------------
# Case data
# -----------------------------
AIRFIELDS = ["London", "Berlin", "Istanbul"]
PORTS = ["Napoli", "Hamburg", "Rotterdam"]
DESTINATIONS = ["Saint Petersburg", "Moscow", "Rostov"]
SOURCES = ["Boston", "Jacksonville"]

NODES = SOURCES + AIRFIELDS + PORTS + DESTINATIONS

coords = {
    "Boston": (-71.06, 42.36), "Jacksonville": (-81.66, 30.33),
    "London": (-0.13, 51.51), "Berlin": (13.40, 52.52), "Istanbul": (28.98, 41.01),
    "Napoli": (14.27, 40.85), "Hamburg": (9.99, 53.55), "Rotterdam": (4.48, 51.92),
    "Saint Petersburg": (30.32, 59.93), "Moscow": (37.62, 55.76), "Rostov": (39.70, 47.24),
}

air_capacity = 150.0
ship_capacity = 240.0
truck_capacity = 16.0
speeds_mph = {"air": 400.0, "sea": 35.0, "land": 60.0}
mph_to_kmph = 1.60934

atlantic_dist = {
    ("Boston", "Berlin"): 7250, ("Boston", "Hamburg"): 8250, ("Boston", "Istanbul"): 8300,
    ("Boston", "London"): 6200, ("Boston", "Rotterdam"): 6900, ("Boston", "Napoli"): 7950,
    ("Jacksonville", "Berlin"): 9200, ("Jacksonville", "Hamburg"): 9800, ("Jacksonville", "Istanbul"): 10100,
    ("Jacksonville", "London"): 7900, ("Jacksonville", "Rotterdam"): 8900, ("Jacksonville", "Napoli"): 9400,
}

eurasia_dist = {
    ("Berlin", "Saint Petersburg"): 1280, ("Hamburg", "Saint Petersburg"): 1880, ("Istanbul", "Saint Petersburg"): 2040,
    ("London", "Saint Petersburg"): 1980, ("Rotterdam", "Saint Petersburg"): 2200, ("Napoli", "Saint Petersburg"): 2970,
    ("Berlin", "Moscow"): 1600, ("Hamburg", "Moscow"): 2120, ("Istanbul", "Moscow"): 1700,
    ("London", "Moscow"): 2300, ("Rotterdam", "Moscow"): 2450, ("Napoli", "Moscow"): 2890,
    ("Berlin", "Rostov"): 1730, ("Hamburg", "Rostov"): 2470, ("Istanbul", "Rostov"): 990,
    ("London", "Rostov"): 2860, ("Rotterdam", "Rostov"): 2760, ("Napoli", "Rostov"): 2800,
}

route_type = {}
for edge in atlantic_dist:
    route_type[edge] = "air" if edge[1] in AIRFIELDS else "sea"
for edge in eurasia_dist:
    route_type[edge] = "air" if edge[0] in AIRFIELDS else "land"

vehicle = {"air": "C-141 Starlifter", "sea": "Transport ship", "land": "Palletized-load truck"}
capacity = {"air": air_capacity, "sea": ship_capacity, "land": truck_capacity}

cost_per_vehicle = {
    ("Boston", "Berlin"): 50000, ("Boston", "Hamburg"): 30000, ("Boston", "Istanbul"): 55000,
    ("Boston", "London"): 45000, ("Boston", "Rotterdam"): 30000, ("Boston", "Napoli"): 32000,
    ("Jacksonville", "Berlin"): 57000, ("Jacksonville", "Hamburg"): 48000, ("Jacksonville", "Istanbul"): 61000,
    ("Jacksonville", "London"): 49000, ("Jacksonville", "Rotterdam"): 44000, ("Jacksonville", "Napoli"): 56000,
    ("Berlin", "Saint Petersburg"): 24000, ("Hamburg", "Saint Petersburg"): 3000, ("Istanbul", "Saint Petersburg"): 28000,
    ("London", "Saint Petersburg"): 22000, ("Rotterdam", "Saint Petersburg"): 3000, ("Napoli", "Saint Petersburg"): 5000,
    ("Berlin", "Moscow"): 22000, ("Hamburg", "Moscow"): 4000, ("Istanbul", "Moscow"): 25000,
    ("London", "Moscow"): 19000, ("Rotterdam", "Moscow"): 5000, ("Napoli", "Moscow"): 5000,
    ("Berlin", "Rostov"): 23000, ("Hamburg", "Rostov"): 7000, ("Istanbul", "Rostov"): 2000,
    ("London", "Rostov"): 4000, ("Rotterdam", "Rostov"): 8000, ("Napoli", "Rostov"): 9000,
}

edges = list(atlantic_dist) + list(eurasia_dist)
edge_df = pd.DataFrame([
    {"From": u, "To": v, "Mode": route_type[(u, v)], "Vehicle": vehicle[route_type[(u, v)]],
     "Distance (km)": atlantic_dist.get((u, v), eurasia_dist.get((u, v))),
     "Speed (mph)": speeds_mph[route_type[(u, v)]], "Cost / vehicle ($)": cost_per_vehicle[(u, v)],
     "Capacity / vehicle (tons)": capacity[route_type[(u, v)]],
     "Cost / ton ($)": cost_per_vehicle[(u, v)] / capacity[route_type[(u, v)]],
     "Time (hours)": atlantic_dist.get((u, v), eurasia_dist.get((u, v))) / (speeds_mph[route_type[(u, v)]] * mph_to_kmph)}
    for u, v in edges
])

requirements = {"Saint Petersburg": 320000.0, "Moscow": 440000.0, "Rostov": 240000.0}


def shortest_path(source, target):
    graph = {n: [] for n in NODES}
    for row in edge_df.itertuples():
        graph[row.From].append((row.To, row._10))  # Time (hours)
    heap = [(0.0, source, [source])]
    seen = {}
    while heap:
        total, node, path = heapq.heappop(heap)
        if node in seen and seen[node] <= total:
            continue
        seen[node] = total
        if node == target:
            return total, path
        for nxt, weight in graph[node]:
            heapq.heappush(heap, (total + weight, nxt, path + [nxt]))
    return math.inf, []


def solve_min_cost():
    n = len(edges)
    c = []
    bounds = []
    for u, v in edges:
        mode = route_type[(u, v)]
        c.append(cost_per_vehicle[(u, v)] / capacity[mode])
        upper = None
        if v == "Saint Petersburg" and mode == "land":
            upper = 0.0
        if v == "Rostov" and mode == "land":
            upper = 2500.0 * truck_capacity
        if (u, v) in [("Berlin", "Rostov"), ("London", "Rostov")]:
            upper = 200.0 * air_capacity
        bounds.append((0.0, upper))

    # Outflow - inflow = net supply at every node.
    A_eq, b_eq = [], []
    net = {node: 0.0 for node in NODES}
    net["Boston"] = 500000.0
    net["Jacksonville"] = 500000.0
    for node in DESTINATIONS:
        net[node] = -requirements[node]
    for node in NODES:
        row = [0.0] * n
        for j, (u, v) in enumerate(edges):
            if u == node:
                row[j] += 1
            if v == node:
                row[j] -= 1
        A_eq.append(row)
        b_eq.append(net[node])

    result = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if not result.success:
        return None, result.message
    flow = result.x
    rows = []
    for (u, v), tons, cost_ton in zip(edges, flow, c):
        if tons > 1e-6:
            mode = route_type[(u, v)]
            rows.append({"From": u, "To": v, "Mode": mode, "Vehicle": vehicle[mode],
                         "Cargo (tons)": tons, "Vehicles / flights": tons / capacity[mode],
                         "Cost / ton ($)": cost_ton, "Route cost ($)": tons * cost_ton})
    return pd.DataFrame(rows), result.fun


def draw_network(highlight_edges=None, title="Available routes"):
    highlight_edges = set(highlight_edges or [])
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {"air": "#2b6cb0", "sea": "#0f766e", "land": "#b45309"}
    node_colors = {**{s: "#be123c" for s in SOURCES}, **{x: "#2563eb" for x in AIRFIELDS},
                   **{x: "#059669" for x in PORTS}, **{x: "#7c3aed" for x in DESTINATIONS}}
    for u, v in edges:
        x1, y1 = coords[u]; x2, y2 = coords[v]
        is_hi = (u, v) in highlight_edges
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#ef4444" if is_hi else colors[route_type[(u, v)]],
                                    lw=3.2 if is_hi else 1.1, alpha=0.95 if is_hi else 0.55,
                                    connectionstyle="arc3,rad=0.03"))
    for node, (x, y) in coords.items():
        ax.scatter(x, y, s=700 if node in DESTINATIONS else 500, c=node_colors[node], edgecolors="white", linewidth=1.5, zorder=3)
        label = "Saint Petersburg" if node == "Saint Petersburg" else node
        ax.text(x, y - 3.1, label, ha="center", va="top", fontsize=8.5, weight="bold")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Longitude (schematic geographic layout)")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.18)
    ax.set_xlim(-90, 48); ax.set_ylim(25, 67)
    plt.tight_layout()
    return fig


st.title("✈️ Aiding Allies: Network Optimization")
st.caption("Case 9-1, Part C — shortest path and minimum-cost flow for moving cargo from the United States to Russia.")

with st.sidebar:
    st.header("Navigation")
    section = st.radio("Open section", ["Network map", "Part B — fastest routes", "Part C — minimum cost", "Data and model"])
    st.divider()
    st.write("**Legend**")
    st.markdown("🔵 Airfield  \n🟢 Port  \n🟣 Russian destination  \n🔴 U.S. source")
    st.caption("All quantities are based on the case data. Costs are converted to dollars per ton for the linear program.")

if section == "Network map":
    st.subheader("a) Network of possible routes")
    st.write("The network is divided into an Atlantic leg from Boston or Jacksonville and a Eurasian leg from a NATO airfield or port to one of the three Russian cities.")
    st.pyplot(draw_network(), clear_figure=True, use_container_width=True)
    st.info("Blue edges are air routes, teal edges are sea routes, and amber edges are truck routes. The map is geographic/schematic; the optimization uses the exact distances in the data table.")

elif section == "Part B — fastest routes":
    st.subheader("b) Shortest-time solution")
    st.write("Because unlimited vehicles are available, the fastest route to each city is found independently with Dijkstra’s shortest-path algorithm. Route time is distance divided by the vehicle speed.")
    rows = []
    all_highlight = []
    for target in DESTINATIONS:
        best = min((shortest_path(source, target) for source in SOURCES), key=lambda x: x[0])
        hours, path = best
        path_edges = list(zip(path[:-1], path[1:]))
        all_highlight.extend(path_edges)
        rows.append({"Destination": target, "Fastest path": " → ".join(path), "Time (hours)": hours, "Time (days)": hours / 24})
    shortest_df = pd.DataFrame(rows)
    st.dataframe(shortest_df.style.format({"Time (hours)": "{:.2f}", "Time (days)": "{:.2f}"}), use_container_width=True, hide_index=True)
    cols = st.columns(3)
    for col, row in zip(cols, rows):
        with col:
            st.metric(row["Destination"], f"{row['Time (hours)']:.2f} hr")
            st.caption(row["Fastest path"])
    st.pyplot(draw_network(all_highlight, "Shortest-time routes highlighted in red"), clear_figure=True, use_container_width=True)
    st.subheader("Fastest time comparison")
    chart_df = shortest_df.set_index("Destination")[["Time (hours)"]]
    st.bar_chart(chart_df, color="#2563eb")

elif section == "Part C — minimum cost":
    st.subheader("c) Minimum-cost flow solution")
    st.write("The model sends exactly 1,000,000 tons from the two U.S. sources to satisfy all three Russian requirements while respecting the special route restrictions.")
    flow_df, objective = solve_min_cost()
    if flow_df is None:
        st.error("The linear program did not solve. Check the deployment logs or verify that scipy is included in requirements.txt.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Minimum total cost", f"${objective:,.0f}")
        k2.metric("Cargo delivered", f"{flow_df['Cargo (tons)'].sum():,.0f} tons")
        k3.metric("Active routes", f"{len(flow_df)}")
        st.success("All requirements are satisfied: Saint Petersburg 320,000 tons; Moscow 440,000 tons; Rostov 240,000 tons.")
        st.dataframe(flow_df.style.format({"Cargo (tons)": "{:,.0f}", "Vehicles / flights": "{:,.0f}", "Cost / ton ($)": "${:,.2f}", "Route cost ($)": "${:,.0f}"}), use_container_width=True, hide_index=True)
        active_edges = list(zip(flow_df["From"], flow_df["To"]))
        st.pyplot(draw_network(active_edges, "Minimum-cost routes highlighted in red"), clear_figure=True, use_container_width=True)
        st.subheader("Cargo by destination")
        by_dest = flow_df.groupby("To", as_index=True)["Cargo (tons)"].sum().reindex(DESTINATIONS).fillna(0)
        st.bar_chart(by_dest, color="#7c3aed")
        st.subheader("Cost contribution by active route")
        cost_chart = flow_df.set_index(flow_df["From"] + " → " + flow_df["To"])[["Route cost ($)"]]
        st.bar_chart(cost_chart, color="#b45309")
        csv = flow_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download minimum-cost route plan (CSV)", csv, "aiding_allies_minimum_cost_plan.csv", "text/csv")

else:
    st.subheader("Data and model assumptions")
    st.write("Use this section as a compact audit trail for classroom discussion or for changing the model inputs in the source code.")
    tabs = st.tabs(["Transportation and routes", "Requirements", "Restrictions", "Mathematical model"])
    with tabs[0]:
        st.dataframe(edge_df.style.format({"Distance (km)": "{:,.0f}", "Speed (mph)": "{:,.0f}", "Cost / vehicle ($)": "${:,.0f}", "Capacity / vehicle (tons)": "{:,.0f}", "Cost / ton ($)": "${:,.2f}", "Time (hours)": "{:.2f}"}), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(pd.DataFrame({"Destination": DESTINATIONS, "Required cargo (tons)": [requirements[x] for x in DESTINATIONS]}).style.format({"Required cargo (tons)": "{:,.0f}"}), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.markdown("- Saint Petersburg can be supplied only by air.\n- From each port, at most 2,500 trucks may be sent to Rostov, equal to 40,000 tons per port.\n- At most 200 flights may go from Berlin to Rostov and 200 from London to Rostov, equal to 30,000 tons on each route.\n- Boston and Jacksonville each have 500,000 tons available.")
    with tabs[3]:
        st.latex(r"\min \sum_{(i,j)\in A} c_{ij}x_{ij}")
        st.latex(r"\sum_j x_{ij}-\sum_j x_{ji}=b_i \quad \forall i")
        st.latex(r"0\le x_{ij}\le u_{ij} \quad \forall (i,j)\in A")
        st.write("Here, xᵢⱼ is tons shipped on route i→j, cᵢⱼ is cost per ton, bᵢ is net supply/demand, and uᵢⱼ is the route capacity implied by the case restrictions.")

st.divider()
st.caption("Built for GitHub + Streamlit Community Cloud. The app uses native Streamlit charts and Matplotlib to avoid the blank-screen deployment problems that can occur with heavier charting dependencies.")
