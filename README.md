# COSMOFlow
**Cosmological parameter inference using gravitational waves and machine learning.**  
COSMOFlow uses **normalising flows** to compute complex probability distributions for cosmological and population parameters inferred from gravitational wave events resulting from compact binary coalescences in the universe. 

### What does COSMOFlow do?
COSMOFlow is designed to compute posterior distributions of the rate of expansion of the universe, specifically the **Hubble constant (H₀)**, using posterior samples from gravitational wave events such as binary black holes. This project leverages Bayesian inference and machine learning to model these distributions efficiently.

![Flow Layout](COSMOFlow/Flow_layout.png)

### Key Features:
- **Efficient Inference**: Use trained normalising flow models to obtain Bayesian posteriors on cosmological parameters such as H₀ in seconds.
- **Scalability**: Applicable to different compact binary coalescence events (e.g., binary black holes, neutron stars) and cosmological models.
- **Use of Prior Information**: Incorporates prior knowledge from galaxy catalogues to improve parameter estimation.

## Paper
For a detailed explanation of the methodology, please refer to the paper associated with this project:  
[**Inferring Cosmological Parameters using Normalizing Flows**](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.123547), published in Physical Review D.

### Abstract:
We present a machine learning approach using normalizing flows for inferring cosmological parameters from gravitational wave events. Our methodology is general to any type of compact binary coalescence event and cosmological model and relies on the generation of training data representing distributions of gravitational wave event parameters. These parameters are conditional on the underlying cosmology and incorporate prior information from galaxy catalogues.  
We demonstrate this approach by inferring the **Hubble constant (H₀)** using binary black hole events detected during the O1, O2, and O3 observational runs of the advanced LIGO/VIRGO detectors. The resulting posterior estimate is 𝐻₀ = 74.51⁢+14.80  
−13.63 km s⁻¹ Mpc⁻¹. Our trained normalizing flow model can compute results in 𝒪(1) second.

---
## Installation

To set up and run COSMOFlow, follow these steps:

**Clone the Repository**:  
   First, you need to download the project to your local machine. Run the following command in your terminal:
   
   ```bash
   git clone https://git.ligo.org/federico.stachurski/cosmoflow.git## Contributing



## Authors and acknowledgment
I would like to thank Christopher Messenger, Martin Hendry, Jessica Irwin, Michael Williams.

