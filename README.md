
### Blockchain Network Performance Analysis

### Methodology

This analysis was conducted through simulations with network sizes of 25, 50, 100, 200, and 300 nodes, exploring 'm' values from 1 to 20. For each configuration, average transaction throughput (transactions per second, tps), latency (seconds), and the number of leader nodes were measured. These metrics serve as indicators of the network's operational efficiency, responsiveness, and resilience, respectively.

### Results Analysis

#### Transaction Throughput and Latency

Across the board, an increase in 'm' generally led to higher throughput up to a certain point, beyond which the gains either plateaued or reversed. This pattern suggests an optimal range of 'm' values that maximize throughput without incurring prohibitive latency. Notably:

* **Throughput**: At smaller network sizes (25 and 50 nodes), lower 'm' values (2-4) were sufficient to achieve relatively high throughput. However, as network size increased, higher 'm' values (8-11) were necessary to maintain or improve throughput efficiency.
* **Latency**: Latency demonstrated an inverse relationship with throughput, where configurations that maximized throughput tended to exhibit lower latency. This relationship underscores the importance of network connectivity in facilitating efficient transaction processing.

#### Leader Node Distribution

The presence of leader nodes, essential for validating transactions and maintaining network integrity, varied with 'm' and network size. An adequate distribution of leader nodes was observed across all network sizes, but the concentration of leaders relative to the total network nodes decreased as 'm' increased. This trend reflects the scale-free nature of the network, where a few nodes become highly connected, potentially centralizing control.

### Figure 1: Graphical Representation of Results

![Figure_1](https://github.com/MohsinRasheed9112/Scale-Free-Network-Analysis/assets/101352612/4dfa7ef0-e28c-4e16-b022-2eff7f6c18b6)


### Discussion

The analysis reveals that 'm' significantly impacts network performance, with an optimal range that shifts based on network size. Smaller networks benefit from lower 'm' values, enabling high throughput and low latency without compromising on the distribution of leader nodes. For larger networks, a higher 'm' value is justified to maintain connectivity and efficiency, though care must be taken to avoid centralization risks.
