# Locked expert responses go here

Empty as of this writing — no experts have been recruited or responded yet
(see `../protocol.md`). When a response is locked per `protocol.md` §3, the
coordinator commits it here as `{expert_code}_responses.csv`, alongside a new
row in `../pre_registration_log.csv`.

Do not hand-edit files in this directory once committed — the whole point of
the pre-registration log is that these files are append-only and their
hashes are independently verifiable (`scripts/score_elicitation.py` checks
this before scoring anything).
