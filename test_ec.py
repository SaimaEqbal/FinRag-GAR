
from src.evidence_checker import EvidenceChecker

checker = EvidenceChecker()

question = (
    "What is the annual increase in interest expense "
    "if LIBOR rises by 100 basis points?"
)

documents = [
    {
        "text": (
            "If LIBOR changes by 100 basis points, "
            "our annual interest expense would change "
            "by $3.8 million."
        )
    }
]

result = checker.check_evidence(question, documents)
print(result)
