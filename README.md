# rnn-lookup

Estado recurrente de tamaño fijo + una acción de "volver a mirar" el contexto
solo cuando hace falta. Experimentos en miniatura (MQAR, Zoology) para ver si
un modelo así iguala a la atención gastando mucho menos.

- Empezar por acá, explicado desde cero: [`docs/guia.md`](docs/guia.md)
- Qué pasó y qué decidimos, con fecha: [`docs/bitacora.md`](docs/bitacora.md)
- Investigación cruda y fuentes: [`docs/notas/`](docs/notas/)
- Guía para trabajar en el repo: [`CLAUDE.md`](CLAUDE.md)

## Arrancar

```bash
git clone --recurse-submodules https://github.com/lucaspecina/rnn-lookup.git
cp scripts/vm.env.example scripts/vm.env           # y completar con los datos de tu VM
scripts/vm.sh start && scripts/vm.sh sync          # VM spot con 2×H100 en Azure
scripts/vm.sh run "bash rnn-lookup/scripts/setup_vm.sh"
scripts/vm.sh run "cd rnn-lookup && source .venv/bin/activate && python -m zoology.launch experiments/mqar/smoke.py"
scripts/vm.sh fetch && scripts/vm.sh stop
```
