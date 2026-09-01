# Instruções de Deploy no Streamlit Cloud

## Pré-requisitos
- Conta GitHub (https://github.com/signup)
- Conta Streamlit Cloud (gratuita, via GitHub)
- Git instalado localmente

## Passo 1: Criar Repositório GitHub

### A. Criar repositório vazio no GitHub
1. Acesse https://github.com/new
2. Nome do repositório: `geopolitica-rag`
3. Descrição: "Plataforma de Simulação Geopolítica - RAG Local"
4. Escolha: **Public** (para outros acessarem)
5. Deixe "Initialize this repository with" desmarcado
6. Clique "Create repository"

### B. Copiar a URL
Após criar, você verá uma URL como:
```
https://github.com/seu-usuario/geopolitica-rag.git
```

## Passo 2: Fazer Push do Código

Abra o terminal na pasta `/home/johannadrianusreismerkx/Downloads` e execute:

```bash
# Inicializar git local
git init

# Adicionar todos os arquivos
git add app_fixed.py requirements.txt README.md .gitignore .streamlit/ geopolitical_conflict_risk_dataset.csv

# Fazer commit
git commit -m "Initial commit: Geopolitical RAG platform with local analysis"

# Definir branch main
git branch -M main

# Adicionar o repositório remoto (substitua URL)
git remote add origin https://github.com/SEU_USUARIO/geopolitica-rag.git

# Fazer push
git push -u origin main
```

### Credenciais Git
Se pedir senha:
- Use seu GitHub username
- Gere um Personal Access Token (PAT) em: https://github.com/settings/tokens
- Use o token como "password"

## Passo 3: Deploy no Streamlit Cloud

1. Acesse https://streamlit.io/cloud
2. Faça login com sua conta GitHub
3. Clique "New app"
4. Preencha:
   - **Repository:** seu-usuario/geopolitica-rag
   - **Branch:** main
   - **Main file path:** app_fixed.py
5. Clique "Deploy"

### Aguarde
- Streamlit Cloud irá:
  1. Clonar seu repositório
  2. Instalar dependências do `requirements.txt`
  3. Rodar o app
  
Esse processo leva 2-5 minutos na primeira vez.

## Passo 4: Compartilhar

Sua URL será:
```
https://seu-usuario-geopolitica-rag.streamlit.app
```

Compartilhe essa URL com qualquer pessoa. Eles podem:
1. Abrir a URL no navegador
2. Anexar um CSV com a estrutura esperada
3. Selecionar os polos de tensão
4. Gerar análises automáticas

## Troubleshooting

### "Module not found"
- Certifique-se de que todas as dependências estão em `requirements.txt`
- Streamlit Cloud pode demorar para instalar; aguarde

### CSV não carrega
- Verifique se o CSV segue a estrutura esperada (26 colunas)
- Teste localmente primeiro: `streamlit run app_fixed.py`

### App muito lento
- Reduza `max_docs` para apenas 100-200 linhas
- O modelo de embedding pode levar tempo na primeira carga

## Próximas Melhorias

- [ ] Adicionar cache persistent para embeddings
- [ ] Criar múltiplas versões de cenários
- [ ] Exportar relatórios em PDF
- [ ] Integração com banco de dados para histórico

