# Master's Thesis on Efficient Long-Haul Truck Driver Routing

This thesis develops a routing algorithm which incorporates maximum driving times without a break and mandatory breaks which truck drivers must comply with in many countries and regions. A query from a start to a target finds a shortest route that stays within those regulatory constraints by navigating to marked parking locations in the road network if necessary. The algorithm is a label-based extension of Dijkstra's algorithm, extended with goal-directed search and a core contraction hierarchy.

The code is written in Rust and can be found [here](https://github.com/maxoe/rust_truck_router).
