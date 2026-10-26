"""Подъём памяти в промпт: место, бюджет, мемоизация, отсутствие поломок.

Три инварианта, которые легко нарушить незаметно:
  * блок стоит в хвосте, перед первой репликой хода (не рвёт префикс-кэш);
  * за один ход блок не меняется, хотя `llm_request` вызывается до 15 раз;
  * сбой поиска не ломает ход — просто нет блока.

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_manager import ContextPool, repair_tool_sequence
from server import config as app_config
from server import memory_context
from server.context import RequestContext, current_request
from server.memory_context import TurnMemory, build_block, render_block
from server.memory_search import EpisodeHit, NodeHit
from server.memory_store import MemoryStore, close_store


def node(node_id: int = 1, name: str = "лимит ElevenLabs",
         text: str = "10000 символов в месяц") -> NodeHit:
    return NodeHit(id=node_id, kind="fact", name=name, text=text, score=5.0)


def episode(eid: int = 1, text: str = "Помни про лимит озвучки") -> EpisodeHit:
    return EpisodeHit(id=eid, ts=1_790_000_000, speaker="human",
                      chat_id="326963375", text=text, score=4.0)


class RenderTest(unittest.TestCase):
    def test_empty_selection_gives_no_block(self) -> None:
        self.assertEqual(render_block([], [], 400), "")

    def test_sections_and_header(self) -> None:
        block = render_block([node()], [episode()], 400)
        self.assertTrue(block.startswith("## Из памяти"))
        self.assertIn("Факты:", block)
        self.assertIn("Из истории:", block)
        self.assertIn("[fact] лимит ElevenLabs", block)
        self.assertIn(f"[{episode().when[:10]}]", block)  # дата, не время

    def test_budget_is_respected(self) -> None:
        nodes = [node(i, f"факт {i}", "текст " * 60) for i in range(1, 9)]
        block = render_block(nodes, [], 120)
        self.assertLessEqual(len(block) // 3, 125)
        self.assertIn("## Из памяти", block)

    def test_only_nodes_still_renders(self) -> None:
        block = render_block([node()], [], 400)
        self.assertNotIn("Из истории:", block)

    def test_too_small_budget_yields_nothing(self) -> None:
        self.assertEqual(render_block([node()], [], 3), "")


class BuildBlockTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "memory.db")
        self.store.upsert_node(
            kind="fact", name="лимит ElevenLabs", name_norm="лимит elevenlabs",
            node_text="free-tier 10000 символов в месяц",
        )
        self.store.add_episode(
            chat_id="326963375", speaker="human", kind="message", ts=1_790_000_000,
            text="Помни: лимит ElevenLabs кончается быстро",
        )

    def tearDown(self) -> None:
        self.store.close()
        close_store()
        self._tmp.cleanup()

    def test_block_contains_fact_and_history(self) -> None:
        block = build_block(self.store, "лимит ElevenLabs")
        self.assertIn("ElevenLabs", block)
        self.assertIn("## Из памяти", block)

    def test_usage_is_recorded(self) -> None:
        build_block(self.store, "лимит ElevenLabs")
        build_block(self.store, "лимит ElevenLabs")
        rows = self.store.connection.execute(
            "SELECT item_type, hits FROM memory_index ORDER BY item_type"
        ).fetchall()
        self.assertTrue(rows)
        self.assertTrue(all(row["hits"] >= 1 for row in rows))
        self.assertIn("node", [row["item_type"] for row in rows])

    def test_near_duplicate_fragments_are_dropped(self) -> None:
        duplicated = ("Одинаковый длинный ответ про рестарт сервиса и его "
                      "последствия для процесса, проверено на живом юните")
        for offset in (0, 60, 120):
            self.store.add_episode(
                chat_id="326963375", speaker="agent", kind="message",
                ts=1_790_000_000 + offset,
                text=duplicated + " подробности: " + ("хвост " * 40)
                     + f"вариант{offset}",
            )
        block = build_block(self.store, "рестарт сервиса")
        self.assertEqual(block.count("Одинаковый длинный ответ"), 1)

    def test_warmup_is_idempotent(self) -> None:
        from server import memory_text

        self.assertTrue(memory_text.warmup())
        self.assertTrue(memory_text.warmup())

    def test_empty_query_gives_nothing(self) -> None:
        self.assertEqual(build_block(self.store, "   "), "")

    def test_search_failure_is_swallowed(self) -> None:
        with mock.patch.object(memory_context, "search_nodes",
                               side_effect=RuntimeError("fts exploded")):
            self.assertEqual(build_block(self.store, "лимит"), "")

    def test_enabled_follows_backend(self) -> None:
        with mock.patch.object(app_config, "MEMORY_BACKEND", "json"):
            self.assertFalse(memory_context.enabled())
        with mock.patch.object(app_config, "MEMORY_BACKEND", "sqlite"):
            self.assertTrue(memory_context.enabled())
        with mock.patch.object(app_config, "MEMORY_BACKEND", " SQLITE "):
            self.assertTrue(memory_context.enabled())


class TurnMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = 0

    def build(self) -> str:
        self.calls += 1
        return f"block-{self.calls}"

    def test_same_turn_builds_once(self) -> None:
        cache = TurnMemory()
        first = cache.get("привет", "1", self.build)
        second = cache.get("привет", "1", self.build)
        self.assertEqual(first, second)
        self.assertEqual(self.calls, 1)
        self.assertEqual(cache.calls, 2)

    def test_new_query_rebuilds(self) -> None:
        cache = TurnMemory()
        cache.get("привет", "1", self.build)
        self.assertEqual(cache.get("другое", "1", self.build), "block-2")

    def test_reset_forces_rebuild(self) -> None:
        cache = TurnMemory()
        cache.get("привет", "1", self.build)
        cache.reset()
        self.assertEqual(cache.get("привет", "1", self.build), "block-2")

    def test_chat_change_rebuilds(self) -> None:
        cache = TurnMemory()
        cache.get("привет", "1", self.build)
        self.assertEqual(cache.get("привет", "2", self.build), "block-2")


class PoolPlacementTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.pool = ContextPool(data_dir=Path(self._tmp.name))
        (Path(self._tmp.name) / "persistent_memory.json").write_text(
            '{"facts": {"оператор": "Андрей"}}', encoding="utf-8"
        )
        self.pool = ContextPool(data_dir=Path(self._tmp.name))
        self.pool.append({"role": "user", "name": "User", "content": "привет"})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def build(self, block: str = "## Из памяти\n- факт"):
        return self.pool.build_llm_messages(
            {"role": "system", "content": "SYS"}, None, memory_block=block
        )

    def test_block_goes_right_before_last_user_message(self) -> None:
        messages = self.build()
        names = [message.get("name") for message in messages]
        self.assertIn("MEMORY", names)
        index = names.index("MEMORY")
        self.assertTrue(messages[index + 1]["content"].endswith("привет"))
        # Профиль идёт раньше блока — блок в хвосте, префикс не тронут.
        self.assertLess(names.index("MASTERMIND"), index)

    def test_empty_block_adds_nothing(self) -> None:
        names = [message.get("name") for message in self.build("")]
        self.assertNotIn("MEMORY", names)
        self.assertEqual(self.pool.memory_block_tokens, 0)

    def test_tokens_are_counted(self) -> None:
        self.build("## Из памяти\n" + "факт " * 100)
        self.assertGreater(self.pool.memory_block_tokens, 0)
        assembled = self.pool.get_assembled_tokens()
        self.assertGreaterEqual(assembled, self.pool.memory_block_tokens)

    def test_block_is_not_stored_in_the_log(self) -> None:
        self.build()
        contents = [entry.content for entry in self.pool._all_entries]
        self.assertNotIn("## Из памяти\n- факт", contents)

    def test_placement_keeps_tool_groups_valid(self) -> None:
        self.pool.append_assistant_message(
            {"content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "fs_read", "arguments": "{}"},
            }]},
            save=False,
            has_tool_calls=True,
        )
        self.pool.append({"role": "tool", "tool_call_id": "c1",
                          "name": "fs_read", "content": "ok"})
        messages = repair_tool_sequence(self.build())
        roles = [message["role"] for message in messages]
        index = [message.get("name") for message in messages].index("MEMORY")
        self.assertEqual(roles[index + 1], "user")
        # Пара tool_calls → tool не разрезана вставкой.
        self.assertEqual(messages[index + 2]["role"], "assistant")
        self.assertEqual(messages[index + 3]["role"], "tool")

    def test_only_profile_means_nowhere_to_insert(self) -> None:
        pool = ContextPool(data_dir=Path(self._tmp.name))
        messages = pool.build_llm_messages(
            {"role": "system", "content": "SYS"}, None, memory_block="- факт"
        )
        self.assertNotIn("MEMORY", [m.get("name") for m in messages])
        self.assertEqual(pool.memory_block_tokens, 0)

    def test_within_turn_assembly_is_stable(self) -> None:
        first = self.build()
        second = self.build()
        self.assertEqual(first, second)


class LastUserTextTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_transport_markers_are_stripped(self) -> None:
        pool = ContextPool(data_dir=self.data_dir)
        pool.append({
            "role": "user",
            "content": "[Time: 26.10.2026, 07:00]"
                       "[MAX messenger | chat_id=1 | user_id=2 | sender=Андрей]\n"
                       "Что там с лимитом?",
        })
        pool.append({"role": "user", "content": "[Tool result from fs_read]: ..."})
        self.assertEqual(pool.last_user_text(), "Что там с лимитом?")

    def test_tool_result_is_skipped(self) -> None:
        pool = ContextPool(data_dir=self.data_dir)
        pool.append({"role": "user", "content": "нормальный вопрос"})
        pool.append({"role": "user", "content": "[Tool result from x]: данные"})
        self.assertEqual(pool.last_user_text(), "нормальный вопрос")

    def test_no_user_message(self) -> None:
        pool = ContextPool(data_dir=self.data_dir)
        self.assertEqual(pool.last_user_text(), "")


class AgentWiringTest(unittest.TestCase):
    """Проверяем связку без создания клиента и без живой data/."""

    def setUp(self) -> None:
        from agent import Agent

        self._tmp = tempfile.TemporaryDirectory()
        self.Agent = Agent
        self.agent = Agent.__new__(Agent)
        self.agent._system_prompt = {"role": "system", "content": "SYS"}
        self.agent._base_prompts = []
        self.agent._use_tools = False
        self.agent.messages = ContextPool(data_dir=Path(self._tmp.name))
        self.agent._turn_memory = TurnMemory()
        self.agent.messages.append({
            "role": "user", "content": "[Time: 26.10.2026, 07:00]\nпривет",
        })

    def tearDown(self) -> None:
        current_request.set(None)
        self._tmp.cleanup()

    def messages(self) -> list[dict]:
        return self.agent._build_messages_for_llm()

    def names(self, messages: list[dict]) -> list[str]:
        return [message.get("name") for message in messages]

    def test_json_backend_adds_no_block(self) -> None:
        with mock.patch.object(app_config, "MEMORY_BACKEND", "json"):
            self.assertNotIn("MEMORY", self.names(self.messages()))

    def test_sqlite_backend_adds_block(self) -> None:
        with mock.patch.object(app_config, "MEMORY_BACKEND", "sqlite"), \
             mock.patch.object(memory_context, "build_block", return_value="- факт"):
            messages = self.messages()
        index = self.names(messages).index("MEMORY")
        self.assertEqual(messages[index]["content"], "- факт")

    def test_block_is_memoised_across_iterations(self) -> None:
        with mock.patch.object(app_config, "MEMORY_BACKEND", "sqlite"), \
             mock.patch.object(memory_context, "build_block",
                               return_value="- факт") as builder:
            self.messages()
            self.messages()
            self.messages()
        self.assertEqual(builder.call_count, 1)

    def test_new_turn_retrieves_again(self) -> None:
        with mock.patch.object(app_config, "MEMORY_BACKEND", "sqlite"), \
             mock.patch.object(memory_context, "build_block",
                               return_value="- факт") as builder:
            self.messages()
            self.agent._turn_memory.reset()
            self.messages()
        self.assertEqual(builder.call_count, 2)

    def test_chat_id_is_passed_to_retrieval(self) -> None:
        current_request.set(RequestContext(request_id="r", source="max", chat_id="77"))
        with mock.patch.object(app_config, "MEMORY_BACKEND", "sqlite"), \
             mock.patch.object(memory_context, "build_block",
                               return_value="") as builder:
            self.messages()
        self.assertEqual(builder.call_args.args[2], "77")


if __name__ == "__main__":
    unittest.main()
