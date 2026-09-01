# ✅ Deploy Checklist — Streamlit Cloud

## Status de Arquivos

- ✅ `app_fixed.py` — App principal (69 KB)
- ✅ `requirements.txt` — Dependências (8 pacotes)
- ✅ `geopolitical_conflict_risk_dataset.csv` — Base padrão (410 KB)
- ✅ `README.md` — Documentação
- ✅ `.streamlit/config.toml` — Configuração
- ✅ `.gitignore` — Arquivos a ignorar no git
- ✅ `DEPLOY_INSTRUCTIONS.md` — Guia passo-a-passo

## Próximos Passos (5-10 minutos)

### 1️⃣ Criar conta GitHub (se não tiver)
- [ ] Vá para https://github.com/signup
- [ ] Confirme o email

### 2️⃣ Criar repositório vazio
- [ ] Acesse https://github.com/new
- [ ] Nome: `geopolitica-rag`
- [ ] Público (Public)
- [ ] Copie a URL

### 3️⃣ Fazer push do código
Na pasta `/home/johannadrianusreismerkx/Downloads`, execute:

```bash
git init
git add app_fixed.py requirements.txt README.md .gitignore .streamlit/ geopolitical_conflict_risk_dataset.csv
git commit -m "Initial commit: Geopolitical RAG platform"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/geopolitica-rag.git
git push -u origin main
```

### 4️⃣ Deploy no Streamlit Cloud
- [ ] Acesse https://streamlit.io/cloud
- [ ] Faça login com GitHub
- [ ] Clique "New app"
- [ ] Selecione: seu-usuario/geopolitica-rag, main, app_fixed.py
- [ ] Clique "Deploy"
- [ ] Aguarde 2-5 minutos

### 5️⃣ Compartilhe
- [ ] Sua URL: https://seu-usuario-geopolitica-rag.streamlit.app
- [ ] Envie para colaboradores/usuários

## O que o Streamlit Cloud Oferece

| Recurso | Plano Gratuito |
|---------|---|
| Apps públicos | ✅ Ilimitado |
| Usuários simultâneos | ✅ Até 10 |
| Reinicializações | ✅ Automático |
| Domínio customizado | ❌ Não |
| SSL | ✅ Sim |
| Custo | ✅ Gratuito |

## Tempo Estimado

- GitHub: 5 min
- Git push: 1 min
- Streamlit deploy: 3-5 min
- **Total: ~10-15 minutos**

## Depois do Deploy

### Para atualizar o código:
```bash
git add .
git commit -m "Update: descrição da mudança"
git push origin main
```
Streamlit Cloud irá redeploy automaticamente.

### Para aumentar performance:
1. Limitar max_docs em `.streamlit/config.toml`
2. Usar `@st.cache_resource` para embeddings
3. Aumentar `client.toolbarMode` se necessário

