# CVnator

[🇧🇷 Leia em português](#-português)

An autonomous, computer-using job application agent. It searches for jobs, reads
them, scores them against your real experience, tailors a CV that stays truthful,
fills in multi-step application forms in a real browser, validates before and after
submitting, and asks you only when it genuinely cannot proceed.

Everything runs on your machine: local models via Ollama, a local database, local
files. Nothing leaves the computer unless you explicitly configure a remote endpoint,
and the code says so out loud when you do.

## What works today

The pipeline runs against a local mock ATS with a real local model and a real
Chromium:

```
CV → Candidate Master Profile → job discovery → job intelligence → match scoring
→ ATS analysis → tailored resume → fact validation → PDF/DOCX render
→ browser agent → multi-step form → upload → pre-submission validation
→ submit → post-submission validation → history → artifact cleanup
```

**Verified end to end** against the mock ATS, with a local model and a real Chromium.
Ten applications across five cold-start runs — separate database, browser profile and
artifact store each time:

```
5 of 5 runs reached a confirmed submission
10 of 10 applications SUBMITTED, 0 failed, 0 needed a human

per run:  3 jobs discovered  →  scored, 1 rejected below threshold
                             →  2 tailored CVs, fact-checked
                             →  5-step form, resume uploaded
                             →  fabricated claims blocked (2-3 per run), each
                                recovered without help
                             →  pre-submission validation → submitted
                             →  confirmed on real evidence (reference number)
                             →  APPLYING → SUBMITTING → SUBMITTED
                             →  temp files deleted, recipe retained

human_intervention_rate  0.000        steps per application  23 / 24 / 24
vision_fallback_rate     0.042        (min / median / max)
tool_retry_rate          0.025
```

Reproduce it with `python scripts/measure_reliability.py --runs 5`.

An in-flight or failed application keeps its resume — only a confirmed submission
triggers cleanup.

Five runs is a small sample against one cooperative ATS. The first attempt at this
measurement lost both applications of its first run, and a later batch still lost one
application in ten; the three defects behind those are fixed and pinned by tests.
Expect to hit cases this has not seen. The failure modes are designed to be
visible (an honest escalation, a recorded failure) rather than silent.

**Scaffolds, clearly marked:** the LinkedIn, Indeed, Glassdoor and Gupy adapters.
Written from the public structure of those sites, **not** validated against
production, and not registered by default — enabling them is an explicit act. See the
note on terms of service below.

200 tests (unit + browser integration against the mock ATS).

## Requirements

- Python 3.12–3.14, Node 20+
- [Ollama](https://ollama.com) running locally
- ~8 GB of VRAM for the default models (the resource manager works within a budget you set)

```bash
ollama pull qwen3.5:9b     # reasoning, resume, translation
ollama pull gemma3:4b      # vision fallback
```

## Setup

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
.venv/Scripts/python -m playwright install chromium
cp .env.example .env
.venv/Scripts/python -m alembic upgrade head
cd frontend && npm install && npm run build && cd ..
```

## Run it

```bash
.venv/Scripts/python -m cvnator.cli serve
```

Then open http://127.0.0.1:8088 and: upload your CV → review the extracted profile →
create a job search → press **Start**.

For UI development, run the API and `npm run dev` in `frontend/` side by side; Vite
proxies `/api` to the Python process.

## Try it safely, against the mock ATS

```bash
.venv/Scripts/python -m cvnator.mockats        # terminal 1
.venv/Scripts/python scripts/run_vertical_slice.py   # terminal 2
```

This exercises the full pipeline without touching anyone's real service.

## The rules this thing follows

These are enforced in code — in the fact validator and the policy engine — not just
asked for in a prompt.

**It will not invent anything about you.** Not a job, a technology, a certification, a
language, a salary, or a number of years. It can rephrase what is true — "created
operational dashboards" may become "developed and maintained dashboards for
operational KPI analysis" — and it refuses anything that adds a fact. Company names,
job titles and dates are copied from your profile; the model is never given a way to
write them.

**It will not defeat a CAPTCHA, an MFA prompt, or a bot check.** It pauses and asks
you. There are no stealth flags and no fingerprint spoofing. If a platform says it is
blocking automation, the agent stops and tells you.

**It will not claim an action succeeded because it ran it.** A click that changed
nothing is a failure. An application is marked submitted only on real evidence — a
confirmation message, a reference number, a confirmation page.

**It will not hide failures.** Failed and abandoned applications are recorded with the
reason.

## Dry run

Dry run is **on by default**. The agent searches, matches, generates CVs and fills
forms, but the final submit is blocked by the policy engine. Turn it off in Settings
when you have seen enough to trust it.

## ATS scores are estimates

No ATS vendor publishes its ranking algorithm, and CVnator does not pretend to
reproduce one. The score is an internal coverage estimate — useful for comparing two
versions of your CV against the same posting, and nothing more. It is not a
probability of getting an interview.

You will also see modest improvements (81% → 82%) more often than dramatic ones. That
is the honesty constraint doing its job: the gap is usually made of skills you do not
have, and the system will not add them.

## A note on real platforms

LinkedIn, Indeed and Glassdoor prohibit automated access in their terms of service.
Using the scaffolded adapters against a real account may get that account restricted.
That is a decision only you can make, which is why those adapters ship disabled and
why there is no code here to help you avoid detection. The mock ATS and the generic
adapter — for career pages and ATS products that do not forbid it — are the paths that
are ready to use.

## Your data

- Your master CV and profile stay in `data/` on your machine.
- Credentials go to the OS credential vault (Windows Credential Manager, macOS
  Keychain, Secret Service) — never to the database, the logs, or a model prompt.
- Logs are filtered for anything that looks like a secret before being written.
- Tailored CVs are temporary by default and deleted once submission is confirmed; the
  recipe to rebuild them is kept.

## Development

```bash
.venv/Scripts/python -m pytest          # tests
.venv/Scripts/python -m ruff check src/ # lint
.venv/Scripts/python -m mypy            # types
cd frontend && npm run typecheck
```

`docs/ARCHITECTURE.md` explains the design decisions and why they are what they are.

---

## 🇧🇷 Português

# CVnator

Um agente autônomo de candidaturas que usa o computador. Ele busca vagas, lê cada uma, pontua contra a sua experiência real, adapta um currículo que continua verdadeiro, preenche formulários de várias etapas num navegador de verdade, valida antes e depois de enviar e só te pergunta quando realmente não consegue seguir.

Tudo roda na sua máquina: modelos locais via Ollama, banco local e arquivos locais. Nada sai do computador a menos que você configure explicitamente um endpoint remoto, e o código avisa em voz alta quando você faz isso.

### O que já funciona

O pipeline roda contra um ATS simulado local, com um modelo local real e um Chromium real:

```
CV → perfil mestre do candidato → descoberta de vagas → leitura da vaga → pontuação de aderência
→ análise de ATS → currículo adaptado → validação de fatos → PDF/DOCX
→ agente no navegador → formulário de várias etapas → upload → validação antes do envio
→ envio → validação depois do envio → histórico → limpeza dos arquivos
```

**Verificado de ponta a ponta** contra o ATS simulado: dez candidaturas em cinco execuções do zero (banco, perfil de navegador e arquivos separados a cada vez). As 5 execuções chegaram a um envio confirmado; as 10 candidaturas foram enviadas, nenhuma falhou e nenhuma precisou de humano. Em cada execução, 3 vagas descobertas, 1 rejeitada abaixo do limite, 2 currículos adaptados e checados, formulário de 5 etapas com upload, 2 a 3 afirmações inventadas bloqueadas e corrigidas sozinhas, e envio confirmado por evidência real (número de protocolo). Métricas: taxa de intervenção humana 0,000, uso do fallback de visão 0,042, taxa de nova tentativa 0,025 e 23/24/24 passos por candidatura (mín./mediana/máx.).

Reproduza com `python scripts/measure_reliability.py --runs 5`. Uma candidatura em andamento ou com falha mantém o currículo; só um envio confirmado dispara a limpeza.

Cinco execuções são uma amostra pequena contra um único ATS cooperativo. A primeira medição perdeu as duas candidaturas da primeira execução, e um lote posterior ainda perdeu uma em dez; os três defeitos por trás disso foram corrigidos e fixados por testes. Espere casos que isto ainda não viu: as falhas foram desenhadas para ser visíveis (uma escalada honesta, uma falha registrada), não silenciosas.

**Esqueletos, claramente marcados:** os adaptadores de LinkedIn, Indeed, Glassdoor e Gupy foram escritos a partir da estrutura pública desses sites, **não** foram validados em produção e não vêm registrados por padrão; ativá-los é um ato explícito. Veja a nota sobre termos de uso abaixo.

200 testes (unitários e de integração no navegador contra o ATS simulado).

### Requisitos

- Python 3.12–3.14 e Node 20+
- [Ollama](https://ollama.com) rodando localmente
- ~8 GB de VRAM para os modelos padrão (o gerenciador de recursos trabalha dentro do limite que você definir)

```bash
ollama pull qwen3.5:9b     # raciocínio, currículo e tradução
ollama pull gemma3:4b      # fallback de visão
```

### Instalação

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
.venv/Scripts/python -m playwright install chromium
cp .env.example .env
.venv/Scripts/python -m alembic upgrade head
cd frontend && npm install && npm run build && cd ..
```

### Executar

```bash
.venv/Scripts/python -m cvnator.cli serve
```

Abra http://127.0.0.1:8088 e: envie seu currículo → revise o perfil extraído → crie uma busca de vagas → aperte **Start**. Para desenvolver a interface, rode a API e o `npm run dev` em `frontend/` lado a lado; o Vite faz proxy de `/api` para o processo Python.

### Testar com segurança, contra o ATS simulado

```bash
.venv/Scripts/python -m cvnator.mockats              # terminal 1
.venv/Scripts/python scripts/run_vertical_slice.py   # terminal 2
```

Isso exercita o pipeline completo sem tocar no serviço real de ninguém.

### As regras que ele segue

Elas são aplicadas no código (no validador de fatos e no motor de políticas), não só pedidas num prompt.

**Ele não inventa nada sobre você.** Nem emprego, nem tecnologia, nem certificação, nem idioma, nem salário, nem número de anos. Ele pode reescrever o que é verdade ("criou dashboards operacionais" pode virar "desenvolveu e manteve dashboards para análise de KPIs operacionais") e recusa qualquer coisa que acrescente um fato. Nomes de empresas, cargos e datas são copiados do seu perfil; o modelo nunca tem como escrevê-los.

**Ele não burla CAPTCHA, MFA nem verificação de bot.** Ele pausa e te pergunta. Não há flags de stealth nem falsificação de fingerprint. Se uma plataforma disser que está bloqueando automação, o agente para e te avisa.

**Ele não diz que uma ação deu certo só porque a executou.** Um clique que não mudou nada é uma falha. Uma candidatura só é marcada como enviada com evidência real: mensagem de confirmação, número de protocolo ou página de confirmação.

**Ele não esconde falhas.** Candidaturas que falharam ou foram abandonadas ficam registradas com o motivo.

### Dry run

O dry run vem **ligado por padrão**. O agente busca, compara, gera currículos e preenche formulários, mas o envio final é bloqueado pelo motor de políticas. Desligue nas configurações quando já tiver visto o suficiente para confiar.

### As notas de ATS são estimativas

Nenhum fornecedor de ATS publica o seu algoritmo de ranking, e o CVnator não finge reproduzir um. A nota é uma estimativa interna de cobertura, útil para comparar duas versões do seu currículo contra a mesma vaga e nada mais. Não é a probabilidade de ser chamado para entrevista. Melhoras modestas (81% → 82%) aparecem mais que as dramáticas: é a regra de honestidade funcionando, porque a diferença costuma ser feita de habilidades que você não tem, e o sistema não vai acrescentá-las.

### Uma nota sobre plataformas reais

LinkedIn, Indeed e Glassdoor proíbem acesso automatizado nos termos de uso. Usar os adaptadores-esqueleto contra uma conta real pode fazer essa conta ser restringida. Essa decisão é só sua, e por isso esses adaptadores vêm desligados e não há código aqui para ajudar a evitar detecção. O ATS simulado e o adaptador genérico (para páginas de carreira e ATS que não proíbem) são os caminhos prontos para uso.

### Seus dados

- O currículo mestre e o perfil ficam em `data/`, na sua máquina.
- Credenciais vão para o cofre do sistema (Gerenciador de Credenciais do Windows, Keychain do macOS, Secret Service), nunca para o banco, os logs ou um prompt de modelo.
- Os logs são filtrados contra qualquer coisa que pareça segredo antes de serem gravados.
- Os currículos adaptados são temporários por padrão e apagados depois do envio confirmado; a receita para refazê-los é mantida.

### Desenvolvimento

```bash
.venv/Scripts/python -m pytest          # testes
.venv/Scripts/python -m ruff check src/ # lint
.venv/Scripts/python -m mypy            # tipos
cd frontend && npm run typecheck
```

O `docs/ARCHITECTURE.md` explica as decisões de design e por que elas são o que são.
