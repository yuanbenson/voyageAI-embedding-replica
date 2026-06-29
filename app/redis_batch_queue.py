from __future__ import annotations

import json
import time
from typing import Any

from redis.asyncio import Redis

from app.queue_models import EmbeddingResultItem, EmbeddingWorkItem
from app.redis_keys import result_queue_key, work_queue_key

_CLAIM_PREFIX_BATCH_LUA = """
local queue_key = KEYS[1]
local max_tokens = tonumber(ARGV[1])
local max_items = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])

local items = redis.call("LRANGE", queue_key, 0, max_items - 1)
local selected = {}
local total_tokens = 0
local remove_count = 0

for _, raw in ipairs(items) do
  local ok, obj = pcall(cjson.decode, raw)
  if not ok then
    -- Malformed head item should not poison the queue forever.
    remove_count = remove_count + 1
  else
    local deadline_ms = tonumber(obj["deadline_ms"] or 0)
    local token_count = tonumber(obj["token_count"] or 0)

    if deadline_ms > 0 and deadline_ms < now_ms then
      -- Drop timed-out requests before they waste GPU work.
      remove_count = remove_count + 1
    elseif #selected == 0 then
      table.insert(selected, raw)
      total_tokens = total_tokens + token_count
      remove_count = remove_count + 1
    elseif total_tokens + token_count <= max_tokens then
      table.insert(selected, raw)
      total_tokens = total_tokens + token_count
      remove_count = remove_count + 1
    else
      break
    end
  end
end

if remove_count > 0 then
  redis.call("LTRIM", queue_key, remove_count, -1)
end

return selected
"""


class RedisBatchQueue:
    def __init__(self, *, redis: Redis, key_prefix: str):
        self._redis = redis
        self._key_prefix = key_prefix
        self._claim_sha: str | None = None

    @classmethod
    def from_url(cls, redis_url: str, key_prefix: str) -> "RedisBatchQueue":
        redis = Redis.from_url(redis_url, decode_responses=True)
        return cls(redis=redis, key_prefix=key_prefix)

    async def close(self) -> None:
        await self._redis.aclose()

    async def ping(self) -> bool:
        return bool(await self._redis.ping())

    async def enqueue_embedding_work(
        self,
        *,
        logical_model: str,
        workload: str,
        item: EmbeddingWorkItem,
    ) -> None:
        key = work_queue_key(self._key_prefix, logical_model, workload)
        await self._redis.rpush(key, item.model_dump_json())

    async def publish_embedding_result(
        self,
        *,
        reply_to: str,
        result: EmbeddingResultItem,
        ttl_seconds: int,
    ) -> None:
        key = result_queue_key(self._key_prefix, reply_to)
        await self._redis.rpush(key, result.model_dump_json())
        await self._redis.expire(key, ttl_seconds)

    async def blpop_embedding_result(
        self,
        *,
        gateway_id: str,
        timeout_seconds: int,
    ) -> EmbeddingResultItem | None:
        key = result_queue_key(self._key_prefix, gateway_id)
        item = await self._redis.blpop(key, timeout=timeout_seconds)
        if item is None:
            return None
        _, raw = item
        return EmbeddingResultItem.model_validate_json(raw)

    async def peek_oldest_embedding_work(
        self,
        *,
        logical_model: str,
        workload: str,
    ) -> EmbeddingWorkItem | None:
        key = work_queue_key(self._key_prefix, logical_model, workload)
        raw = await self._redis.lindex(key, 0)
        if raw is None:
            return None
        try:
            return EmbeddingWorkItem.model_validate_json(raw)
        except Exception:
            # Let the Lua claim script remove malformed head items.
            return None

    async def claim_embedding_batch(
        self,
        *,
        logical_model: str,
        workload: str,
        max_tokens: int,
        max_items: int,
        now_ms: int | None = None,
    ) -> list[EmbeddingWorkItem]:
        key = work_queue_key(self._key_prefix, logical_model, workload)
        now_ms = now_ms if now_ms is not None else current_time_ms()
        raw_items = await self._eval_claim_script(
            keys=[key],
            args=[str(max_tokens), str(max_items), str(now_ms)],
        )

        parsed: list[EmbeddingWorkItem] = []
        for raw in raw_items:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            parsed.append(EmbeddingWorkItem.model_validate_json(raw))
        return parsed


    async def list_embedding_work_raw(
        self,
        *,
        logical_model: str,
        workload: str,
        max_items: int | None = None,
    ) -> list[str]:
        key = work_queue_key(self._key_prefix, logical_model, workload)
        stop = -1 if max_items is None else max(0, max_items - 1)
        raw_items = await self._redis.lrange(key, 0, stop)
        return [item.decode("utf-8") if isinstance(item, bytes) else item for item in raw_items]

    async def queue_length(self, *, logical_model: str, workload: str) -> int:
        key = work_queue_key(self._key_prefix, logical_model, workload)
        return int(await self._redis.llen(key))

    async def _eval_claim_script(self, *, keys: list[str], args: list[str]) -> list[Any]:
        if self._claim_sha is None:
            self._claim_sha = await self._redis.script_load(_CLAIM_PREFIX_BATCH_LUA)
        try:
            return await self._redis.evalsha(self._claim_sha, len(keys), *keys, *args)
        except Exception as exc:
            # Redis loses loaded scripts on restart. Fall back once to EVAL.
            if "NOSCRIPT" not in str(exc).upper():
                raise
            self._claim_sha = await self._redis.script_load(_CLAIM_PREFIX_BATCH_LUA)
            return await self._redis.evalsha(self._claim_sha, len(keys), *keys, *args)


def current_time_ms() -> int:
    return int(time.time() * 1000)


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)
