# Plataforma de Simulação Geopolítica — RAG Local

## Descrição

Sistema de análise de risco geopolítico baseado em CSV local, com:
- Seleção de dois polos de tensão (país/ator principal vs. rival)
- Derivação automática de parâmetros a partir dos dados
- Análise RAG com base de conhecimento local (sem chaves externas)
- Projeção de escalada de conflito
- Relatórios estratégicos com evidências

## Estrutura do CSV Base

O CSV deve conter as seguintes colunas:
- **Identificação:** country, region, month
- **Econômicos:** gdp_growth_pct, inflation_rate, unemployment_rate, food_price_index, energy_dependency_pct, trade_dependency_rival_pct
- **Políticos:** political_stability_index, media_freedom_score, regime_type, election_cycle
- **Defesa:** military_expenditure_pct_gdp, arms_imports_index, border_disputes_count, sanctions_active, cyber_attack_incidents, foreign_troops_present, last_conflict_year
- **Sociais:** protest_events_last_3m, rolling_protest_avg_6m, refugee_outflow_thousands, social_media_sentiment
- **Alvo:** instability_score, conflict_escalation_6m

## Como Usar Localmente

```bash
# Clone ou baixe o repositório
git clone https://github.com/seu-usuario/geopolitica-rag.git
cd geopolitica-rag

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Rode o app
streamlit run app_fixed.py
```

O app abrirá em `http://localhost:8501`.

## Deploy no Streamlit Cloud

1. **Prepare o repositório GitHub:**
   - Faça push do código, `requirements.txt` e `.streamlit/config.toml`
   - Inclua um CSV de exemplo ou instrua o upload

2. **Acesse Streamlit Cloud:**
   - Vá para https://streamlit.io/cloud
   - Faça login com sua conta GitHub

3. **Crie um novo app:**
   - Clique em "New app"
   - Selecione o repositório, branch e arquivo (`app_fixed.py`)
   - Clique "Deploy"

4. **Compartilhe a URL:**
   - Seu app ficará em: `https://seu-usuario-geopolitica-app.streamlit.app`

## Funcionalidades

- **Painel de Simulação:** Contexto derivado automaticamente dos polos selecionados
- **Relatório Estratégico:** Análise com fontes (linhas do CSV)
- **KB / Debug:** Teste direto de consultas na base de conhecimento

## Notas

- O app não requer chaves de API externas (OpenAI, etc.)
- Os embeddings e índices são calculados localmente
- A base CSV é anexada no início (upload ou arquivo padrão)
