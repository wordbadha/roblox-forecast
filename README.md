# Roblox Up-and-Coming forecast

A Python 3 program (standard library only, no network) that reads the
Up-and-Coming chart history at `$MAKEFUN_INPUT_DIR` and writes one
`forecast.json` to `$MAKEFUN_OUTPUT_DIR`: a probability for each game at
each of the top 10 places, plus a top-10 guess, for the 72 hours after
entries close.

The model: a game's rank drifts down the chart over time (estimated per
rank band from read pairs 12–96 hours apart), its future rank is
Laplace-distributed around last rank + drift × horizon with a scale that
grows with the horizon, and the forecast is the time-average of that
density over the window's hourly slots. Games are named at a place only
while their probability beats that place's unnamed share.

Practice score on the 10 reads of 8–10 November 2026: log score **−34.26**
(house baseline −43.51), typical probability 0.033 per place, 7.0 top-10
hits per read.

Referral: https://make.fun/r/czd88s7a
