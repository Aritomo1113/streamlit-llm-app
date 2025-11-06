# 手順4-4: メインファイルから環境変数の読み込み
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage

# --- 専門家の定義 ---
EXPERT_A_NAME = "歴史学者"
EXPERT_A_ROLE = "あなたは世界中の歴史に精通した歴史学者です。質問に対して、正確で詳細な歴史的背景や事実に基づいて回答してください。情報の出所や時代背景も示し、専門家としての深みのある解説を心がけてください。"

EXPERT_B_NAME = "ファイナンシャルプランナー"
EXPERT_B_ROLE = "あなたは個人の資産運用や税制に詳しいファイナンシャルプランナーです。質問者の状況に基づき、実現可能でリスクを考慮した具体的な資産形成のアドバイスをしてください。専門用語は分かりやすく説明し、信頼できる情報源に基づいた回答をしてください。"


# --- 処理を担う関数 ---
def get_llm_response(input_text: str, selected_expert: str) -> str:
    """
    LLMからの回答を取得する関数

    Args:
        input_text (str): ユーザーからの入力テキスト
        selected_expert (str): ラジオボタンで選択された専門家の名前

    Returns:
        str: LLMからの回答
    """
    # 選択に応じてシステムメッセージを決定
    if selected_expert == EXPERT_A_NAME:
        system_role = EXPERT_A_ROLE
    elif selected_expert == EXPERT_B_NAME:
        system_role = EXPERT_B_ROLE
    else:
        # デフォルト設定 (ありえないが念のため)
        system_role = "あなたは親切なAIアシスタントです。"

    # LangChainのテンプレート構築
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_role)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt, 
        human_message_prompt
    ])
    
    # プロンプトのフォーマット
    formatted_prompt = chat_prompt.format_messages(text=input_text)

    # LLMのインスタンス化 (Pythonのバージョンは3.11を想定)
    # LangChainのOpenAIモデルは、自動で環境変数 OPENAI_API_KEY を参照します。
    # model_nameにはご自身のOpenAIアカウントで利用可能なモデルを指定してください。
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo") 

    # LLMの呼び出し
    response = llm.invoke(formatted_prompt)

    # 回答テキストを返す
    return response.content


# --- Streamlit UI構築 ---

# Webアプリの概要や操作方法をユーザーに明示
st.title("👨‍🏫 専門家選択式 LLM アプリ")
st.write("このアプリでは、ラジオボタンで選択した専門家のロールに基づき、LLMが回答を行います。")
st.markdown("---")


# ラジオボタンで専門家を選択
selected_expert = st.sidebar.radio(
    "1. 専門家を選択してください",
    (EXPERT_A_NAME, EXPERT_B_NAME)
)
st.sidebar.markdown(f"**選択中の専門家**: **{selected_expert}**")
st.sidebar.markdown("---")

# メイン画面に入力フォームを配置
user_input = st.text_area(
    "2. 質問を入力してください:",
    placeholder=f"例: {selected_expert}に質問したい内容をここに入力..."
)

# 送信ボタン
if st.button("質問を送信"):
    if user_input:
        st.info("回答を生成中です...しばらくお待ちください。")
        
        # 関数を呼び出しLLMの回答を取得
        try:
            llm_response = get_llm_response(user_input, selected_expert)
            
            # 結果の表示
            st.success(f"🤖 **{selected_expert}** からの回答:")
            st.markdown(llm_response)
        
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.warning("OpenAI APIキーが正しく設定されているか、または課金が有効になっているか確認してください。")
            
    else:
        st.warning("質問を入力してから送信ボタンを押してください。")

# アプリ実行方法: 仮想環境を有効にした状態で `streamlit run app.py` をコマンドラインで実行します。