import asyncio
import os

# ActionModel / RegisteredAction / ActionRegistry が必要なら、以下のようにまとめてインポート
# from browser_use.controller.registry.views import (
#     ActionModel,
#     RegisteredAction,
#     ActionRegistry
# )

from browser_use import Agent, Controller, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from langchain_openai import ChatOpenAI

# Controller を初期化
controller = Controller()

@controller.registry.action('take_screenshot', requires_browser=True)
async def take_screenshot(browser: BrowserContext, path: str = "screenshot.png"):
    """
    現在のタブのスクリーンショットを撮るアクション。
    """
    # フォルダが存在しない場合は作成
    os.makedirs("screenshot", exist_ok=True)

    page = await browser.get_current_page()
    screenshot_path = os.path.join("screenshot", path)
    await page.screenshot(path=screenshot_path)
    print(f"スクリーンショットを {screenshot_path} に保存しました。")

    return {
        "extracted_content": f"{screenshot_path} に保存完了",
        "include_in_memory": True
    }

async def main():
    # ブラウザの設定（例として headless=False, keep_open=True）
    browser = Browser(
        config=BrowserConfig(
            headless=False,
            # keep_open=True,
        )
    )

    # タスク例
    target_url = "https://moji.onl.jp/"
    task = f"指定されたURL '{target_url}' にアクセスし、入力欄に「あいうえお」と入力し、文字数のボタンを押してください。また、その結果をスクリーンショットで保存してください。画像の名前は、'moji_result.png'としてください。"

    # LLMのセットアップ
    model = ChatOpenAI(model='gpt-4o')

    # Agent に controller を渡す（controller に定義したアクションが呼び出されます）
    agent = Agent(
        task=task,
        llm=model,
        browser=browser,
        controller=controller,
    )

    await agent.run()
    await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
