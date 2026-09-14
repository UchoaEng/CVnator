"""A vocabulary of technology terms used to detect *unsupported* skill claims.

This is not an attempt to know every technology in existence. It is a detector: if a
generated resume mentions a term from this list that the candidate's master profile
does not support, that is a fabricated skill claim and the fact validator blocks it.
Terms outside this list are handled by the structural guarantees in the optimizer
(identity fields are copied, never generated) and by bullet-level grounding checks.
"""

from __future__ import annotations

import re

TECH_VOCABULARY: frozenset[str] = frozenset(
    {
        # languages
        "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "golang",
        "rust", "ruby", "php", "scala", "kotlin", "swift", "r", "matlab", "perl", "bash",
        "shell", "powershell", "sql", "plsql", "t-sql", "vba", "dart", "elixir", "haskell",
        "lua", "groovy", "objective-c", "clojure", "julia", "cobol", "fortran", "assembly",
        # data / analytics
        "pandas", "numpy", "scipy", "polars", "dask", "spark", "pyspark", "hadoop", "hive",
        "kafka", "airflow", "dbt", "snowflake", "redshift", "bigquery", "databricks",
        "tableau", "power bi", "powerbi", "looker", "metabase", "superset", "qlik",
        "grafana", "kibana", "matplotlib", "seaborn", "plotly", "d3.js", "d3", "excel",
        "google sheets", "sas", "stata", "spss", "alteryx", "talend", "informatica",
        "flink", "beam", "luigi", "prefect", "dagster", "great expectations",
        # ml / ai
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost", "lightgbm",
        "catboost", "huggingface", "transformers", "langchain", "llamaindex", "mlflow",
        "opencv", "nltk", "spacy", "onnx", "cuda", "ollama", "vllm",
        # databases
        "postgresql", "postgres", "mysql", "mariadb", "sqlite", "oracle", "sql server",
        "mongodb", "cassandra", "dynamodb", "redis", "elasticsearch", "opensearch", "neo4j",
        "clickhouse", "influxdb", "timescaledb", "couchbase", "firebase", "supabase",
        "duckdb", "cockroachdb",
        # cloud / infra
        "aws", "azure", "gcp", "google cloud", "digitalocean", "heroku", "vercel", "netlify",
        "kubernetes", "k8s", "docker", "podman", "terraform", "pulumi", "ansible", "chef",
        "puppet", "vagrant", "helm", "argocd", "jenkins", "gitlab ci", "github actions",
        "circleci", "travis", "teamcity", "bamboo", "openshift", "nomad", "consul", "vault",
        "cloudformation", "lambda", "ec2", "s3", "rds", "eks", "ecs", "fargate", "sagemaker",
        "cloudwatch", "athena", "glue", "kinesis", "sqs", "sns", "step functions",
        # observability
        "prometheus", "datadog", "new relic", "splunk", "opentelemetry", "jaeger", "zipkin",
        "sentry", "loki", "tempo", "elk", "nagios", "zabbix", "pagerduty", "dynatrace",
        "appdynamics", "victoriametrics", "thanos", "mimir",
        # web / frameworks
        "react", "vue", "angular", "svelte", "next.js", "nextjs", "nuxt", "remix", "astro",
        "django", "flask", "fastapi", "express", "nestjs", "spring", "spring boot", "rails",
        "laravel", "symfony", "asp.net", ".net", "dotnet", "graphql", "rest", "grpc",
        "websocket", "html", "css", "sass", "scss", "tailwind", "bootstrap", "jquery",
        "node.js", "nodejs", "deno", "bun", "webpack", "vite", "babel", "redux",
        # tooling / practices
        "git", "github", "gitlab", "bitbucket", "jira", "confluence", "trello", "asana",
        "notion", "figma", "linux", "unix", "windows server", "macos", "nginx", "apache",
        "rabbitmq", "activemq", "celery", "pytest", "junit", "jest", "cypress", "selenium",
        "playwright", "cucumber", "sonarqube", "postman", "swagger", "openapi",
        "scrum", "kanban", "agile", "safe", "itil", "devops", "sre", "ci/cd", "tdd", "bdd",
        "microservices", "etl", "elt", "data warehouse", "data lake", "data mesh",
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "business intelligence", "data visualization", "statistics", "a/b testing",
        "sap", "salesforce", "servicenow", "workday", "oracle erp", "totvs", "protheus",
    }
)

# Multi-word terms must be matched before single tokens, longest first.
_SORTED_TERMS: tuple[str, ...] = tuple(
    sorted(TECH_VOCABULARY, key=len, reverse=True)
)

_TERM_PATTERN = re.compile(
    "|".join(rf"(?<![\w.+#]){re.escape(term)}(?![\w+#])" for term in _SORTED_TERMS),
    re.IGNORECASE,
)


def find_tech_terms(text: str) -> set[str]:
    """Return the lowercase technology terms mentioned in ``text``."""
    if not text:
        return set()
    return {match.group(0).lower() for match in _TERM_PATTERN.finditer(text)}


#: Terms that are effectively the same skill, so a profile listing one supports the other.
ALIASES: dict[str, set[str]] = {
    "postgres": {"postgresql"},
    "postgresql": {"postgres"},
    "golang": {"go"},
    "go": {"golang"},
    "k8s": {"kubernetes"},
    "kubernetes": {"k8s"},
    "sklearn": {"scikit-learn"},
    "scikit-learn": {"sklearn"},
    "powerbi": {"power bi"},
    "power bi": {"powerbi"},
    "nodejs": {"node.js"},
    "node.js": {"nodejs"},
    "nextjs": {"next.js"},
    "next.js": {"nextjs"},
    "gcp": {"google cloud"},
    "google cloud": {"gcp"},
    "d3": {"d3.js"},
    "d3.js": {"d3"},
    "dotnet": {".net"},
    ".net": {"dotnet"},
}


def expand_aliases(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for term in terms:
        expanded |= ALIASES.get(term, set())
    return expanded
