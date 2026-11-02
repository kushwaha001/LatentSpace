# Data card: mock transit backend (`data/backend/`)

The deterministic backend behind the `transit_agent` tools. Built by
`python data/backend/build_backend.py`. The JSON files are generated from that script, so edit
the script, not the JSON.

## What is real and what is synthetic

| Data | Real or synthetic | Notes |
|---|---|---|
| Indian Railways station names and codes (45 stations, `rail_stations.json`) | **Real** | Codes such as `CSMT`, `NDLS`, `PUNE`, `PRYJ`, `KOP`. Please check them before publishing. |
| Metro station names and line order: Mumbai Line 1 and Line 3 (Aqua), Pune Purple and Aqua, Delhi Yellow, Blue and Airport Express (`metro.json`) | **Real names**. Some lines are a **contiguous subset** of the real line | Delhi Blue stops at Yamuna Bank, Yellow starts at Vishwavidyalaya. Interchanges come from shared names (Marol Naka, Civil Court, Rajiv Chowk, New Delhi, Dwarka Sector 21). |
| Metro station ids (`mum_andheri`, `del_rajiv_chowk`, ...) | **Synthetic** | In-repo identifiers, not official operator codes. |
| Hindi and Marathi station names and aliases | Model-written | They need native-speaker review, like the eval text. |
| Trains (`trains.json`): numbers `S2xxxx`, names, stops, timetables, running days, distances | **SYNTHETIC** | They do not describe real services. Every train name ends in "(synthetic)", and every number starts with `S` so it cannot be mistaken for a real 5-digit train number. |
| Fares (`fares.json`): per-km rates, reservation charges, metro slabs | **SYNTHETIC** | Not the real IRCTC or metro fares. |
| Seat availability | **SYNTHETIC** | A fixed hash of (train, class, date). |
| PNRs (`bookings.json`, `84xxxxxxxx`) and new bookings (`86xxxxxxxx`) | **SYNTHETIC** | |

Do not use any timetable, fare or availability value from this backend for real travel.

## Behaviour

- **Names in any script.** `normalize_name` NFC-normalises, case-folds, drops nukta,
  punctuation and words such as "junction", "station", "स्टेशन", "जंक्शन", "मेट्रो", and maps
  Devanagari digits. A name then matches the station's English name, its Hindi and Marathi
  names, the code and the romanised aliases. A city-level name ("Mumbai", "दिल्ली") returns
  every candidate station, so the agent should ask which one.
- **Not handled:** inflected forms (Marathi "पुण्याहून", Hindi "पुणे से"). The agent must pass
  the base name. Misspellings beyond the alias list are not handled either (there is no fuzzy
  match).
- **State.** Bookings, cancellations and complaints change the `MockBackend` instance. Call
  `transit_agent.reset_backend()` before each episode.
- **Service ids.** `S20000-CSMT-PUNE` (a train between two of its stops) and
  `MET-<City>-<from_id>-<to_id>` (a metro route). `get_fare` and `book_ticket` take either.
