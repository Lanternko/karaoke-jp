# Holistic Singing Benchmark

| Dataset | Lang | System | Cond. | COn | COnP | COnPOff | Gran. | MAE | median | hit% | thr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kiritan | ja | CE+CTC | zero-shot | .860 | .652 | .492 | - | - | - | - |  |
|  |  | GAME | zero-shot | .862 | .644 | .502 | - | - | - | - |  |
|  |  | ROSVOT | zero-shot | .414 | .290 | .190 | - | - | - | - |  |
|  |  | MMS-JA | zero-shot | - | - | - | phone | 0.218s | 0.055s | 46.7% | <=50ms |
|  |  | MMS_FA | zero-shot | - | - | - | phone | 0.112s | 0.043s | 56.0% | <=50ms |
|  |  | SOFA ! | contaminated | - | - | - | phone | 0.018s | 0.007s | 89.3% | <=50ms |
| itako | ja | CE+CTC | zero-shot | .828 | .509 | .374 | - | - | - | - |  |
|  |  | GAME | zero-shot | .824 | .494 | .400 | - | - | - | - |  |
|  |  | MMS-JA | zero-shot | - | - | - | phone | 0.113s | 0.055s | 46.7% | <=50ms |
|  |  | MMS_FA | zero-shot | - | - | - | phone | 0.078s | 0.048s | 52.0% | <=50ms |
| mir-st500 | zh | CE+CTC | in-domain | .779 | .728 | .554 | - | - | - | - |  |
|  |  | GAME | zero-shot | .732 | .655 | .411 | - | - | - | - |  |
|  |  | ROSVOT | zero-shot | .108 | .093 | .063 | - | - | - | - |  |
| jamendolyrics | en+ | MMS_FA | zero-shot | - | - | - | word | 0.233s | - | 94.5% | PCO@.3 |
| custom-gold | ja | MMS-JA | canonical | - | - | - | line | 0.037s | 0.031s | 100.0% | <=250ms |
|  |  | SOFA+island | zero-shot | - | - | - | line | 0.177s | 0.046s | 83.0% | <=250ms |
|  |  | classic | legacy | - | - | - | line | 1.903s | 1.240s | 0.0% | <=250ms |