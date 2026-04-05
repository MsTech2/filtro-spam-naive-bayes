🛡️ Detector de Spam com Inteligência Artificial - Naive Bayes

Para este projeto utilizei o algoritmo **Naive Bayes** para classificar mensagens de texto como "SPAM" ou "Mensagem Real". 
Exemplo prático de como a Inteligência Artificial pode ser usada para filtrar comunicações indesejadas com base em padrões de palavras.

---

## 🚀 Teste na prática

Como este é um ambiente de visualização, convido você pode testar o código em tempo real seguindo estes passos:

1. Acesse o **[Google Colab](https://colab.research.google.com/)**.
2. Clique em **"Novo Notebook"**.
3. Copie o código abaixo e cole na célula que aparecer:

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --- ETAPA 1: O DATASET  ---
data = {
    'text': [
        # --- MENSAGENS REAIS (HAM) ---
        'Oi, tudo bem? Vamos almoçar hoje?',
        'Esqueci minha chave no escritório, você viu?',
        'Segue o arquivo da reunião de amanhã',
        'Oi mãe, chego em casa às 19h para o jantar',
        'Preciso de ajuda com o projeto de faculdade',
        'A chave está na recepção do prédio',
        'Pode me passar o contato do João?',
        'Parabéns pelo seu aniversário, muitas felicidades!',
        'Você vai na aula hoje à noite?',
        'O relatório ficou pronto, confira por favor',

        # --- MENSAGENS DE SPAM ---
        'GANHE DINHEIRO FACIL AGORA CLIQUE NO LINK',
        'VOCÊ GANHOU UM PREMIO EXCLUSIVO DE 1000 REAIS',
        'PROMOÇÃO IMPERDIVEL COMPRE JA COM DESCONTO',
        'Sua conta foi bloqueada, acesse o link urgente',
        'Ganhe um pix de 500 reais agora clicando aqui',
        'Oferta exclusiva: leve dois e pague um hoje',
        'Clique aqui para liberar seu brinde grátis',
        'Urgente: atualize seus dados cadastrais no link',
        'Você foi selecionado para uma vaga, clique para saber mais',
        'CUIDADO: Alerta de segurança na sua conta, acesse agora'
    ],
    'label': ['ham']*10 + ['spam']*10
}

df = pd.Series(data['label']).map({'ham': 0, 'spam': 1})
texts = data['text']

# --- ETAPA 2: PREPARAÇÃO (Vetorização) ---
cv = CountVectorizer()
X = cv.fit_transform(texts)
y = df

# Dividindo em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ETAPA 3: TREINAMENTO (Naive Bayes) ---
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# --- ETAPA 4: TESTE COM UMA FRASE ---
def prever_mensagem(frase):
    frase_vetorizada = cv.transform([frase])
    predicao = modelo.predict(frase_vetorizada)
    resultado = "SPAM 🚨" if predicao[0] == 1 else "MENSAGEM REAL ✅"
    print(f"Frase: '{frase}' -> Resultado: {resultado}")

# TESTANDO AGORA COM PALAVRAS QUE O MODELO JÁ VIU NO TREINO:
print("--- TESTANDO O CLASSIFICADOR ---")
prever_mensagem("Oi mãe, tudo certo? Vamos almoçar?") # Esta deve dar REAL
prever_mensagem("GANHE DINHEIRO AGORA CLIQUE NO LINK") # Esta deve dar SPAM

# Espaço para o usuário visitante digitar
mensagem_do_usuario = input("Digite a mensagem para testar: ")
prever_mensagem(mensagem_do_usuario)
