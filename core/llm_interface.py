"""
Lightweight LLM adapter utilities.

Provides a single `get_completion()` helper which accepts either:
- an object exposing `generate(prompt, temperature=...)` (existing codepaths)
- a LangGraph graph id / client (attempts common call patterns)
- a callable that accepts `prompt` and `temperature`
- LangChain/LangGraph runnables with invoke()

This centralizes LangGraph integration while keeping backward compatibility.
"""

import importlib
import json
from typing import Any


def _normalize(res: Any) -> str:
	"""
	Normalize various LLM response formats to string.
	
	Handles:
	- Plain strings
	- Dicts with 'output', 'text', or full JSON
	- Any other type converted to string
	"""
	if isinstance(res, str):
		return res
	if isinstance(res, dict):
		output = res.get("output") or res.get("text")
		if output:
			return output if isinstance(output, str) else str(output)
		return json.dumps(res)
	return str(res)



def get_completion(llm: Any, prompt: str, temperature: float = 0.2, timeout: int = None) -> str:
	"""Return a string completion using a variety of LLM interfaces.

	Args:
		llm: An LLM object, graph id, client, dict, or callable.
		prompt: The text prompt to send.
		temperature: Sampling temperature (note: may be ignored by some graphs).
		timeout: Optional timeout in seconds for execution.

	Returns:
		String completion text.

	Raises:
		RuntimeError: if no supported interface is found or all attempts fail.
		
	Note:
		Temperature parameter may be ignored by compiled LangGraph graphs
		or other interfaces that override this at compile time. Always
		test temperature behavior with your specific graph.
	"""
	last_error = None

	# ===== 1. Try generate() method (existing integrations) =====
	gen = getattr(llm, "generate", None)
	if callable(gen):
		try:
			return _normalize(gen(prompt=prompt, temperature=temperature))
		except Exception as e:
			last_error = e

	# ===== 2. Try invoke() method (LCEL Runnable, LangGraph compiled) =====
	invoke = getattr(llm, "invoke", None)
	if callable(invoke):
		try:
			# Try with dict input first (LangGraph pattern)
			res = invoke({"prompt": prompt, "temperature": temperature})
			return _normalize(res)

		except Exception as e1:
			try:
				# Fallback: try with "input" key (some graphs)
				res = invoke({"input": prompt})
				return _normalize(res)

			except Exception as e2:
				try:
					# Final fallback: raw string input (RunnableLambda style)
					res = invoke(prompt)
					return _normalize(res)

				except Exception as e3:
					last_error = e3




	# ===== 3. Try LangGraph Client patterns =====
	try:
		lg = importlib.import_module("langgraph")
	except Exception:
		lg = None

	if lg is not None:
		# Determine graph id / reference
		graph_id = None
		if isinstance(llm, dict):
			graph_id = llm.get("graph_id") or llm.get("graph")
		elif isinstance(llm, str):
			graph_id = llm
		else:
			graph_id = getattr(llm, "graph_id", None) or getattr(llm, "graph", None)

		# Try locating a client
		Client = getattr(lg, "Client", None) or getattr(lg, "LangGraphClient", None)
		client = None
		try:
			if Client:
				client = Client()
		except Exception:
			pass

		# Try calling client methods
		if client and graph_id:
			for method_name in ("run_graph", "run", "execute", "call"):
				fn = getattr(client, method_name, None)
				if callable(fn):
					try:
						res = fn(graph_id, inputs={"prompt": prompt, "temperature": temperature})
						return _normalize(res)
					except Exception as e:
						last_error = e
						continue

		# Try module-level helpers
		for module_fn_name in ("run_graph", "run", "execute"):
			fn = getattr(lg, module_fn_name, None)
			if callable(fn) and graph_id:
				try:
					res = fn(graph_id, inputs={"prompt": prompt, "temperature": temperature})
					return _normalize(res)
				except Exception as e:
					last_error = e
					continue

	# ===== 4. Try callable fallback (positional + keyword) =====
	if callable(llm):
		try:
			# Try keyword args first
			return llm(prompt=prompt, temperature=temperature)
		except TypeError:
			try:
				# Fallback: positional args
				return llm(prompt, temperature)
			except Exception as e:
				try:
					# Last resort: positional with no temp
					return llm(prompt)
				except Exception as e2:
					last_error = e2

	# ===== Failure with context =====
	msg = "No suitable LLM interface available: provide an object with `generate()` or `invoke()`, or a LangGraph graph/client"
	if last_error:
		msg += f"\nLast error: {type(last_error).__name__}: {str(last_error)}"
	
	raise RuntimeError(msg)