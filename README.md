# DecoFormer:Enhancing CrossFormer's Performance through DLinear Integration

**Decoformer** mainly addresses Time Series Forecasting(TSF) problem based on [CrossFormer](https://github.com/Thinklab-SJTU/Crossformer) (ICLR2023). 
Considering that TSF is time-related, 
whereas Crossformer's design of Cross-Time Stage(CTS) applies time-invariant Multi-Attention 
Mechanism, this work abandons CTS and integrates another model called Decomposition Linear 
(also noted as *Dlinear*). Consequently, the result shows that while maintaining comparable accuracy, 
Decoformer can handle TSF with a simpler structure. 
Furthermore, it has better performance on Long-term Forcasting.

