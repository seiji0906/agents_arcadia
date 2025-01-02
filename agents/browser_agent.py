import logging
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from browser_use import Agent as BrowserUseAgent, Controller, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from agents.base_agent import BaseAgent

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BrowserAgent(BaseAgent):
    def __init__(self, llm=None, tools=None, task: str = ""):
        super().__init__(llm, tools)
        self.task = task
        self.controller = Controller()
        self.browser = Browser(config=BrowserConfig(headless=True))
        logging.info("BrowserAgent 初期化完了")

    async def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        logging.info("BrowserAgent.runを開始します。")
        # browser_use の Agent クラスを使って任意のタスクを実行
        agent = BrowserUseAgent(
            task=self.task,
            llm=self.llm,
            browser=self.browser,
            controller=self.controller
        )
        try:
            # ブラウザを開き、タスクを実行する（例として run()）
            await agent.run()
            return "BrowserAgent: タスクが完了しました。"
        except Exception as e:
            logging.error(f"BrowserAgent 実行中にエラーが発生しました: {e}")
            return "BrowserAgent 実行中にエラーが発生しました。"

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        logging.info("BrowserAgent.arunを開始します。")
        # 同期的に run() を呼ぶか、必要に応じて非同期化
        return await self.run(input, config)