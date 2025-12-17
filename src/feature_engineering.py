import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["delivery_gap"] = (
        df["planned_story_points"] - df["completed_story_points"]
    ) / (df["planned_story_points"] + 1)

    df["workload_pressure"] = (
        df["avg_work_hours"] / 40
    ) + (df["meeting_hours"] / 20)

    df["quality_instability"] = (
        df["bug_count"] +
        df["reopened_issues"] +
        df["deployment_failures"]
    ) / (df["test_coverage_pct"] + 1)

    df["scope_pressure"] = (
        df["scope_change_requests"] +
        df["backlog_growth_rate"] * 10
    )

    df["execution_risk_index"] = (
        df["dependency_blockers"] +
        abs(df["velocity_change_pct"]) +
        df["code_churn_pct"]
    )

    df["team_health_index"] = (
        df["attrition_risk_score"] +
        df["communication_delay_hours"] / 10 +
        df["unplanned_leave_days"]
    )

    df["time_pressure"] = df["milestone_slippage_days"] / (
        df["days_remaining"] + 1
    )

    df.drop(columns=["completed_story_points"], inplace=True, errors="ignore")

    return df
