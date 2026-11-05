
class AgenticRetrievalController:
    """Centralized, rule-based decisions for the FinRAG-GAR pipeline."""

    ACTIONS = {
        "REWRITE_QUERY",
        "RETRIEVE_MORE",
        "GENERATE_ANSWER",
        "CORRECT_ANSWER",
        "VERIFY_ANSWER",
        "ACCEPT_ANSWER",
        "ABSTAIN",
    }

    def __init__(self, max_retries=2):
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer.")

        self.max_retries = max_retries

    @staticmethod
    def _decision(action, reason, **extra):
        result = {"action": action, "reason": reason}
        result.update(extra)
        return result

    @staticmethod
    def _is_valid_query(query, original_query):
        return (
            isinstance(query, str)
            and bool(query.strip())
            and query.strip().lower() != original_query.strip().lower()
        )

    def select_rewrite(self, original_query, rewrite_result):
        """
        Select one valid candidate using deterministic rules.

        Prefer the first valid candidate. If it is unavailable,
        use the second valid candidate. No additional LLM call is made.
        """
        if not isinstance(original_query, str) or not original_query.strip():
            return self._decision(
                "ABSTAIN",
                "The original query is empty or invalid.",
            )

        if not isinstance(rewrite_result, dict):
            return self._decision(
                "ABSTAIN",
                "The rewrite result is invalid.",
            )

        candidates = rewrite_result.get("candidates", [])

        if not isinstance(candidates, list):
            return self._decision(
                "ABSTAIN",
                "The rewrite candidates must be a list.",
            )

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            rewritten_query = candidate.get("query")
            model = candidate.get("model", "unknown")

            if self._is_valid_query(rewritten_query, original_query):
                selected_query = rewritten_query.strip()

                print("========== CONTROLLER DECISION ==========")
                print(f"Selected model: {model}")
                print(f"Selected query: {selected_query}")
                print("==========================================\n")

                return self._decision(
                    "RETRIEVE_MORE",
                    "Selected a valid rewritten query for retrieval.",
                    selected_query=selected_query,
                    selected_model=model,
                )

        return self._decision(
            "RETRIEVE_MORE",
            "No valid rewrite was generated. Continue with the original query.",
            selected_query=original_query,
            selected_model="original",
        )

    def decide(
        self,
        evidence_result=None,
        verification_result=None,
        retry_count=0,
        query_changed=False,
        rewrite_used=False,
        answer=None,
    ):
        """Determine the next action from evidence and verification results."""

        if (
            not isinstance(retry_count, int)
            or isinstance(retry_count, bool)
            or retry_count < 0
        ):
            return self._decision("ABSTAIN", "Invalid retry count.")

        # 1. Handle answer verification.
        if verification_result is not None:
            if not isinstance(verification_result, dict):
                return self._decision(
                    "ABSTAIN", "Invalid verification result."
                )

            if verification_result.get("status") != "ok":
                return self._decision(
                    "ABSTAIN", "Answer verification failed."
                )

            claims = verification_result.get("claim_assessments", [])
            supported = verification_result.get("supported") is True

            all_claims_supported = (
                isinstance(claims, list)
                and bool(claims)
                and all(
                    isinstance(claim, dict)
                    and claim.get("label") == "supported"
                    for claim in claims
                )
            )

            if supported and all_claims_supported:
                return self._decision(
                    "ACCEPT_ANSWER",
                    "All assessed claims are supported by the evidence.",
                )

            corrected_answer = verification_result.get("corrected_answer")

            if isinstance(corrected_answer, str) and corrected_answer.strip():
                return self._decision(
                    "CORRECT_ANSWER",
                    "The draft answer requires correction.",
                    corrected_answer=corrected_answer.strip(),
                )

            if retry_count < self.max_retries:
                return self._decision(
                    "RETRIEVE_MORE",
                    "The answer is not fully supported; more evidence is needed.",
                )

            return self._decision(
                "ABSTAIN",
                "The answer could not be verified within the retry limit.",
            )

        # 2. Verify a non-empty draft answer.
        if answer is not None:
            if isinstance(answer, str) and answer.strip():
                return self._decision(
                    "VERIFY_ANSWER",
                    "Verify the draft answer against the selected evidence.",
                )

            return self._decision(
                "ABSTAIN",
                "The draft answer is empty or invalid.",
            )

        # 3. Validate the evidence assessment.
        if not isinstance(evidence_result, dict):
            return self._decision(
                "ABSTAIN", "No valid evidence assessment was provided."
            )

        if evidence_result.get("status") != "ok":
            return self._decision(
                "ABSTAIN", "Evidence assessment failed."
            )

        assessment_1 = evidence_result.get("assessment_1")
        assessment_2 = evidence_result.get("assessment_2")

        # Require both assessments when dual-model results are provided.
        if assessment_1 is not None or assessment_2 is not None:
            assessments = [assessment_1, assessment_2]

            valid_assessments = [
                item for item in assessments
                if (
                    isinstance(item, dict)
                    and item.get("status") == "ok"
                    and isinstance(item.get("sufficient"), bool)
                )
            ]

            if not valid_assessments:
                return self._decision(
                    "ABSTAIN",
                    "Both evidence assessments failed."
                )

            both_sufficient = any(
                item["sufficient"] for item in valid_assessments
            )
        else:
            # Backward compatibility with a single top-level assessment.
            both_sufficient = evidence_result.get("sufficient") is True

        conflicts = evidence_result.get("potential_conflicts", [])

        if not isinstance(conflicts, list):
            return self._decision(
                "ABSTAIN", "The evidence conflict data is invalid."
            )

        blocking_conflict = any(
            isinstance(conflict, dict)
            and conflict.get("classification")
            in {"confirmed", "unresolved"}
            for conflict in conflicts
        )

        if (
            both_sufficient
        ):
            return self._decision(
                "GENERATE_ANSWER",
                "Evidence is sufficient and has no blocking conflicts.",
            )

        # 4. Do not rewrite more than once per pipeline run.
        if not rewrite_used:
            return self._decision(
                "REWRITE_QUERY",
                "Evidence is insufficient; generate two rewrite candidates.",
            )

        # 5. After rewriting, permit bounded retrieval retries, then abstain.
        if retry_count < self.max_retries:
            return self._decision(
            "GENERATE_ANSWER",
            "Retrieval attempts exhausted. Generate the best "
            "evidence-supported answer, including partial information.",
        )

        return self._decision(
            "GENERATE_ANSWER",
            "Retrieval attempts are exhausted. Generate the best "
            "partial answer supported by the available evidence.",
        )
