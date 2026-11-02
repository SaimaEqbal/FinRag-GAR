from src.agentic_controller import AgenticRetrievalController

controller = AgenticRetrievalController(max_retries=2)

result = controller.select_rewrite(
    original_query="What was Microsoft's revenue in fiscal year 2023?",
    rewrite_result={
        "candidates": [
            {
                "model": "model_1",
                "query": "Microsoft total revenue fiscal year 2023",
            },
            {
                "model": "model_2",
                "query": "Microsoft consolidated income statement revenue 2023",
            },
        ]
    },
)

print(result)