# Optimal Routing AI: Genetic Algorithm vs Dijkstra 🚀

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, interactive, and beautiful demonstration of **Multi-Objective Evolutionary Algorithms (MOEA)** solving NP-Hard QoS-constrained network routing problems across a highly complex, multi-attribute ISP backbone. 

This project proves mathematically and visually why greedy minimum-cost algorithms fall apart in the real world, and how evolutionary intelligence can dynamically navigate strict Service Level Agreements (SLAs).

---

## 🌟 The Core Problem: Why Routing is Hard

In traditional networking, finding a path from Point A to Point B is solved instantly by **Dijkstra's Algorithm**. Dijkstra is mathematically perfect at finding the shortest or cheapest path. 

However, modern networks (like global ISP backbones) do not operate on a single metric. They operate on **Quality of Service (QoS) constraints**. A network operator typically wants to:
1. **Minimize Cost** (Fiber transit costs money).
2. **Minimize Latency** (Financial trading packets must arrive under 50ms).
3. **Maximize Reliability** (Enterprise data cannot afford packet drops).

When you introduce constraints (e.g., *"Find me the cheapest path, but it MUST be under 100ms latency"*), the routing problem transitions from simple polynomial time to **NP-Hard**. Dijkstra evaluates only one variable at a time; it is "blind" to secondary constraints when making its greedy hops. It will happily route a critical low-latency packet onto an ultra-cheap, but painfully slow, satellite link.

### 🧬 The Solution: NSGA-II
This project introduces a custom-built **Non-dominated Sorting Genetic Algorithm II (NSGA-II)** as the solver. Instead of looking step-by-step, it evaluates hundreds of entire paths simultaneously, evolving them over generations to find the absolute best paths satisfying all constraints concurrently.

---

## ✨ Key Technical Features

- **True Multi-Objective Optimization (MOO)**: Instead of arbitrary weighted-sum equations (which break easily), the engine evaluates independent Cost, Latency, and Reliability objectives.
- **Fast Non-Dominated Sorting (Pareto Fronts)**: The population of routes is mathematically sorted into 3D Pareto Fronts, ensuring that the algorithm explores all optimal trade-offs.
- **Constrained Domination Tunneling**: Routes that violate SLAs (like a max latency constraint) aren't instantly deleted. They are mathematically graded on their "Margin of Violation". This allows the genetic algorithm to safely tunnel through temporarily 'invalid' regions of the map to uncover premium subpaths on the other side.
- **Crowding Distance**: Prevents the AI from converging too early by mathematically calculating the density of routes and forcing offspring to explore sparse, unknown regions of the network.
- **Complex Topologies**: Features a dense Indian ISP backbone network with over 40 nodes/links, deliberately injected with "Satellite Traps" (ultra-cheap but 150ms+ latency links) to test the algorithm's resilience.

---

## 🎯 The "Satellite Trap" Scenario

To perfectly illustrate the engine's power, imagine you are routing data from **Delhi (DEL)** to **Chennai (CHN)**. The enterprise SLA mandates a strict **Max Latency of 100ms**.

1. **Dijkstra (Greedy approach)**: 
   Looks at the map and sees a direct Satellite link that costs mere pennies ($3). It greedily chooses this path. However, the satellite introduces 140ms of latency! Dijkstra violently violates the SLA.
2. **The Genetic Algorithm (Evolutionary approach)**: 
   Analyzes the Pareto front and recognizes the latency barrier. It completely bypasses the satellite trap, intelligently stitching together a slightly more expensive ($25), but ultra-fast (40ms) Fiber mesh through Central India. **Payload delivered safely within SLA parameters.**

*(You can see a beautiful, interactive visual animation of this exact scenario by opening `about.html` in your web browser!)*

---

## 🚀 Quick Start Guide

### 1. Local Installation

Ensure you have Python 3.8+ installed on your machine.

```bash
# Clone the repository
git clone https://github.com/your-username/Routing_GA.git
cd Routing_GA

# Install the required dependencies
pip install -r requirements.txt
```

### 2. Running the Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The application will spin up vividly on your local web browser (`http://localhost:8501`). 

### 3. Deploying to Streamlit Cloud

Because a `requirements.txt` is included, deploying is effortless:
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository, and set the **Main file path** to `app.py`.
4. Click Deploy. Streamlit handles the entire virtual environment naturally!

*(Note: Do not use `start.py` as your cloud entry point, as it will trigger port conflicts).*

---

## 🖥️ How to Use the Dashboard

When you open the web app, you will be greeted by the **Control Panel** and the **Network Graph**.

1. **Routing Setup**: Select your Source and Destination cities from the sidebar. 
2. **Set QoS Constraints**: Adjust the sliders for **Max Latency Allowed** and **Min Reliability Allowed**. Set these tight to force Dijkstra into failing!
3. **GA Hyperparameters**: (Optional) Tweak the DNA! Adjust Population Size, Max Generations, and Mutation Rates to see how it affects convergence speed.
4. **Execute Simulation**: Hit the main button. The Python backend will run both Dijkstra and the full GA evolutionary cycle in milliseconds.
5. **Analyze the Results**: 
   - View the detailed table comparing the GA vs Dijkstra.
   - Inspect the **Convergence Plot**: See exactly how the genetic algorithm learned, optimized costs, and maintained diversity over time.

---

## 📁 Repository Structure

- `app.py`: The frontend UI, sidebar controls, event handling, and Plotly visualization generation. Set this as your main entry point for deployment.
- `routing_engine.py`: The computational heart. Contains the NetworkX graph architecture, physical topological layout (`build_network()`), Dijkstra variants, and the deep NSGA-II GA math.
- `requirements.txt`: Python package dependency list for easy cloud deployment.
- `about.html`: A beautiful standalone interactive HTML file explaining the "Satellite Trap" via a dynamic animated packet-routing demo.
- `start.py`: A lightweight local entry script if running outside a native terminal.

---

## 📝 Open Source License

This software is provided "as is", without warranty of any kind. 
Licensed under the [MIT License](LICENSE). Feel free to fork, mess with the DNA, and evolve it further!
