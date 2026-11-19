# Sobrevivência de carreiras na NFL

Código, dados e resultados do Trabalho de Conclusão de Curso *Sobrevivência na NFL* (Bacharelado em Estatística, UFSCar, 2026), sobre a longevidade de carreiras de quarterbacks (QB), running backs (RB) e wide receivers (WR) na NFL, temporadas de 2012 a 2024. O trabalho combina:

- **análise exploratória** e curvas de Kaplan–Meier;
- **inferência** com modelos de Cox, Aalen e Cox–Aalen, com covariáveis dependentes do tempo;
- **predição** da saída da liga com validação walk-forward, comparando Kaplan–Meier, Cox penalizado, GBSA e Random Survival Forest (RSF).

**A tese está em [`Escrita Final/texto/main.pdf`](Escrita%20Final/texto/main.pdf).**

## Reprodutibilidade

- `python run_all.py` refaz toda a análise a partir de `DataBase/yearly_player_stats_offense.csv`, em 48 passos, e regrava todos os resultados e figuras.
- O selo da execução de referência está em `reprodutibilidade/EXECUTION_CHECKPOINT.json`, com o hash de cada resultado.
- Ao final, `reprodutibilidade/audit_escrita_vs_pipeline.py` confere os números da tese contra os resultados. O relatório está em `reprodutibilidade/AUDITORIA_ESCRITA_VS_PIPELINE.md`.
- `reprodutibilidade/build_tese.py` recompila a tese a partir das fontes em `Escrita Final/texto/` e das figuras geradas.

## Organização

| Pasta ou arquivo | Conteúdo |
|---|---|
| `run_all.py`, `pipeline/` | Orquestrador e os 48 passos numerados da análise |
| `DataBase/` | Base de entrada e tabelas de qualidade da base |
| `Análise Exploratória de Dados/` | Figuras e tabelas da análise exploratória |
| `Inferência/` | Código dos modelos Cox, Aalen e Cox–Aalen; resultados inferenciais e de sensibilidade |
| `Predição/` | Código dos modelos preditivos; em `pacote_default/`, a comparação com os hiperparâmetros padrão das bibliotecas, sem ajuste (não são os números da tese) |
| `documentos/` | Estudos complementares citados nos apêndices: cada `estudo_*` tem o script e os resultados. Os resultados preditivos finais da tese estão em `documentos/estudo_modelos_finais/resultados/` |
| `Escrita Final/` | Tese: PDF e fontes LaTeX (`texto/`) e figuras (`figuras/`) |
| `reprodutibilidade/` | Manifesto, selo da execução, auditorias e compilação da tese |
| `requirements/` | Versões fixadas das dependências Python e R |

## Desenho do estudo

- **Evento.** A saída da liga é aproximada pelo último registro anual observado; não é uma data física de aposentadoria. Jogadores ainda ativos em 2024 são censurados. As variáveis `observed_career_lag_*` resumem o desempenho acumulado antes de cada temporada, dentro da janela analisada.
- **Inferência.** O Cox inferencial é ajustado sem penalização, com o método de Efron para os empates. O Aalen (`survival::aareg`) trata as saídas de cada ano em bloco. O Cox–Aalen usa `timereg::cox.aalen`; antes do ajuste, as saídas anuais empatadas são deslocadas para dentro do próprio intervalo anual.
- **Predição.** A avaliação é walk-forward: em cada origem, o treino respeita o corte temporal, e a busca de hiperparâmetros é aninhada ao treino daquela origem. Imputação e padronização são ajustadas em cada treino.

O Cox inferencial (sem penalização) e o Cox preditivo (penalizado) são especificações diferentes; não os misture ao interpretar coeficientes ou desempenho.

## Resultados preditivos

C de Harrell médio nos quatro folds de teste (temporadas 2018 a 2021). Fonte: `documentos/estudo_modelos_finais/resultados/resumo_walkforward_2018_2021.csv`.

| Modelo | QB | RB | WR |
|---|---:|---:|---:|
| Kaplan–Meier (referência) | 0,500 | 0,500 | 0,500 |
| Cox penalizado | 0,705 | 0,704 | 0,704 |
| GBSA | 0,676 | 0,665 | 0,687 |
| RSF | 0,709 | 0,661 | 0,693 |

O melhor modelo por posição é o RSF no QB e o Cox penalizado nos RB e WR.

## Ambiente

- **Python 3.12.** A execução de referência usou Python 3.12.3, com as versões fixadas em `requirements/requirements_avaliacao.txt`.
- **R 4.3.3.** A execução de referência usou `timereg` 2.0.7 e `survival` 3.5-8. O arquivo `requirements/requirements_R.txt` lista `survival` 3.8.6, que não foi a versão usada: para reproduzir os valores-p do Aalen, instale a 3.5-8.
- `Rscript` deve estar no `PATH`. Para recompilar a tese, também `pdflatex` e `bibtex`.

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/requirements_avaliacao.txt
Rscript -e 'install.packages(c("timereg", "survival", "jsonlite"))'
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements/requirements_avaliacao.txt
```

`install.packages` instala a versão mais recente de cada pacote R; para fixar as versões listadas em `requirements/requirements_R.txt`, use `remotes::install_version`. Versões diferentes podem produzir pequenas diferenças numéricas, sobretudo no GBSA.

## Execução

```bash
python run_all.py
```

Executa os 48 passos em ordem, depois de apagar os resultados gerados, e regrava tudo. Figuras geradas que a tese não usa são apagadas no fim.

Opções:

- `--resume`: continua a partir do primeiro passo pendente, sem apagar resultados.
- `--fast`: pula os estudos pesados (ajuste, leitura e furos). Serve apenas para iterar; não equivale à execução de referência.
- `--release-audit`: revalida uma execução concluída contra o selo.

## Verificação

```bash
python reprodutibilidade/reproducibility_audit.py --static
python reprodutibilidade/audit_escrita_vs_pipeline.py --strict
```

- `reproducibility_audit.py --static`: confere o repositório contra o manifesto (`reprodutibilidade/REPRODUCIBILITY_MANIFEST.json`) sem executar a análise.
- `audit_escrita_vs_pipeline.py --strict`: confere os números da tese contra os resultados e retorna código de saída 1 se houver divergência.

## Compilação da tese

```bash
python reprodutibilidade/build_tese.py --build-dir /caminho/para/pasta_nova
```

Compila `Escrita Final/texto/` com as figuras de `Escrita Final/figuras/` numa pasta fora do repositório. No Windows, use `powershell -File reprodutibilidade\build_tese.ps1`.

## Escopo

O alcance dos resultados está discutido na seção *Limitações* do capítulo de conclusão da tese. Em resumo: os dados são anuais, a saída é reconhecida pela ausência nas temporadas seguintes, e a censura em 2024 limita o horizonte observado; os resultados inferenciais são associações, e não efeitos causais, e as listas finais de covariáveis vêm dos mesmos dados da estimação; o Aalen (`aareg`) e o Cox–Aalen (`timereg`) tratam os empates de formas diferentes, de modo que seus resultados não são equivalentes.
