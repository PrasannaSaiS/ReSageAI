"""
extractor.py — Resume file parsing and text normalisation.
Single Responsibility: extract clean text from PDF / DOCX / DOC files.
Open/Closed: add new formats by extending _extract_* without modifying existing ones.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Optional extraction libraries ─────────────────────────────────────────────
try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore

try:
    import docx
except ImportError:
    docx = None  # type: ignore

try:
    import textract
except ImportError:
    textract = None  # type: ignore

# ── Technical term normalisation map ──────────────────────────────────────────
# OCP: extend this dict to support new terms without touching extraction logic.
TECH_NORMALIZATION: dict[str, str] = {
    r"\b(fast\s*api)\b": "FastAPI",
    r"\b(py\s*torch)\b": "PyTorch",
    r"\b(num\s*py)\b": "NumPy",
    r"\b(git\s*hub\s*actions)\b": "GitHub Actions",
    r"\b(git\s*hub)\b": "GitHub",
    r"\b(git\s*lab)\b": "GitLab",
    r"\b(bit\s*bucket)\b": "BitBucket",
    r"\b(dev\s*ops)\b": "DevOps",
    r"\b(java\s*script)\b": "JavaScript",
    r"\b(type\s*script)\b": "TypeScript",
    r"\b(mongo\s*db)\b": "MongoDB",
    r"\b(postgres\s*ql)\b": "PostgreSQL",
    r"\b(word\s*press)\b": "WordPress",
    r"\b(j\s*query)\b": "jQuery",
    r"\b(tensor\s*flow)\b": "TensorFlow",
    r"\b(open\s*cv)\b": "OpenCV",
    r"\b(power\s*bi)\b": "PowerBI",
    r"\b(sql\s*alchemy)\b": "SQLAlchemy",
    r"\b(spring\s*boot)\b": "SpringBoot",
    r"\b(node\s*j\s*s)\b": "Node.js",
    r"\b(node\s*dot\s*js)\b": "Node.js",
    r"\b(vue\s*j\s*s)\b": "Vue.js",
    r"\b(react\s*j\s*s)\b": "ReactJS",
    r"\b(web\s*pack)\b": "Webpack",
    r"\b(sqlite)\b": "SQLite",
    r"\b(mysql)\b": "MySQL",
    r"\b(dynamo\s*db)\b": "DynamoDB",
    r"\b(cosmos\s*db)\b": "CosmosDB",
    r"\b(cassandra)\b": "Cassandra",
    r"\b(elastic\s*search)\b": "Elasticsearch",
    r"\b(graph\s*ql)\b": "GraphQL",
    r"\b(apache\s*spark)\b": "Apache Spark",
    r"\b(microsoft)\b": "Microsoft",
    r"\b(sales\s*force)\b": "Salesforce",
    r"\b(quick\s*books)\b": "QuickBooks",
    r"\b(wire\s*shark)\b": "Wireshark",
    r"\b(virtual\s*box)\b": "VirtualBox",
    r"\b(virtual\s*env)\b": "virtualenv",
    r"\b(proto\s*buf)\b": "Protobuf",
    r"\b(grpc)\b": "gRPC",
    r"\b(rest\s*ful)\b": "RESTful",
    r"\b(oauth)\b": "OAuth",
    r"\b(auth\s*0)\b": "Auth0",
    r"\b(key\s*cloak)\b": "Keycloak",
    r"\b(active\s*mq)\b": "ActiveMQ",
    r"\b(rabbit\s*mq)\b": "RabbitMQ",
    r"\b(snowflake)\b": "Snowflake",
    r"\b(big\s*query)\b": "BigQuery",
    r"\b(red\s*shift)\b": "Redshift",
    r"\b(fargate)\b": "Fargate",
    r"\b(cloud\s*formation)\b": "CloudFormation",
    r"\b(terraform)\b": "Terraform",
    r"\b(jenkins)\b": "Jenkins",
    r"\b(circle\s*ci)\b": "CircleCI",
    r"\b(argo\s*cd)\b": "ArgoCD",
    r"\b(prom\s*ql)\b": "PromQL",
    r"\b(prometheus)\b": "Prometheus",
    r"\b(grafana)\b": "Grafana",
    r"\b(data\s*dog)\b": "Datadog",
    r"\b(new\s*relic)\b": "New Relic",
    r"\b(splunk)\b": "Splunk",
    r"\b(log\s*stash)\b": "Logstash",
    r"\b(kibana)\b": "Kibana",
    r"\b(postman)\b": "Postman",
    r"\b(swagger)\b": "Swagger",
    r"\b(open\s*api)\b": "OpenAPI",
    r"\b(apigee)\b": "Apigee",
    r"\b(kong)\b": "Kong",
    r"\b(traefik)\b": "Traefik",
    r"\b(envoy)\b": "Envoy",
    r"\b(istio)\b": "Istio",
    r"\b(consul)\b": "Consul",
    r"\b(nomad)\b": "Nomad",
    r"\b(vault)\b": "Vault",
    r"\b(packer)\b": "Packer",
    r"\b(vagrant)\b": "Vagrant",
    r"\b(vm\s*ware)\b": "VMware",
    r"\b(prox\s*mox)\b": "Proxmox",
    r"\b(podman)\b": "Podman",
    r"\b(open\s*shift)\b": "OpenShift",
    r"\b(kustomize)\b": "Kustomize",
    r"\b(flux\s*cd)\b": "FluxCD",
    r"\b(spin\s*naker)\b": "Spinnaker",
    r"\b(team\s*city)\b": "TeamCity",
    r"\b(sonar\s*cloud)\b": "SonarCloud",
    r"\b(sonar\s*qube)\b": "SonarQube",
    r"\b(snyk)\b": "Snyk",
    r"\b(veracode)\b": "Veracode",
    r"\b(fortify)\b": "Fortify",
    r"\b(check\s*marx)\b": "Checkmarx",
    r"\b(black\s*duck)\b": "Black Duck",
    r"\b(tail\s*wind\s*css)\b": "Tailwind CSS",
    r"\b(tail\s*wind)\b": "Tailwind CSS",
    r"\b(bootstrap)\b": "Bootstrap",
    r"\b(material\s*ui)\b": "Material-UI",
    r"\b(chakra\s*ui)\b": "Chakra UI",
    r"\b(daisy\s*ui)\b": "daisyUI",
    r"\b(svelte)\b": "Svelte",
    r"\b(solid\s*js)\b": "SolidJS",
    r"\b(preact)\b": "Preact",
    r"\b(alpine\s*js)\b": "Alpine.js",
    r"\b(back\s*bone\s*js)\b": "Backbone.js",
    r"\b(knock\s*out)\b": "Knockout",
    r"\b(web\s*components)\b": "Web Components",
    r"\b(shadow\s*dom)\b": "Shadow DOM",
    r"\b(custom\s*elements)\b": "Custom Elements",
    r"\b(next\s*js)\b": "Next.js",
    r"\b(nuxt\s*js)\b": "Nuxt.js",
    r"\b(gatsby)\b": "Gatsby",
    r"\b(remix)\b": "Remix",
    r"\b(astro)\b": "Astro",
    r"\b(express)\b": "Express",
    r"\b(fastify)\b": "Fastify",
    r"\b(nest\s*js)\b": "NestJS",
    r"\b(web\s*3)\b": "Web3",
    r"\b(solidity)\b": "Solidity",
    r"\b(macos)\b": "macOS",
    r"\b(sci\s*py)\b": "SciPy",
    r"\b(sea\s*born)\b": "Seaborn",
    r"\b(stats\s*models)\b": "statsmodels",
    r"\b(hugging\s*face)\b": "Hugging Face",
    r"\b(lang\s*chain)\b": "LangChain",
    r"\b(ollama)\b": "Ollama",
    r"\b(milvus)\b": "Milvus",
    r"\b(pine\s*cone)\b": "Pinecone",
    r"\b(chroma\s*db)\b": "ChromaDB",
    r"\b(weaviate)\b": "Weaviate",
    r"\b(qdrant)\b": "Qdrant",
    r"\b(faiss)\b": "Faiss",
    r"\b(neo\s*4j)\b": "Neo4j",
    r"\b(orient\s*db)\b": "OrientDB",
    r"\b(graph\s*db)\b": "GraphDB",
    r"\b(arango\s*db)\b": "ArangoDB",
}


# ── Private format extractors (OCP: add new ones freely) ──────────────────────
def _extract_pdf(path: Path) -> tuple[str, bool]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed.")
    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        has_photo = any(page.images for page in pdf.pages)
    return text, has_photo


def _extract_docx(path: Path) -> tuple[str, bool]:
    if docx is None:
        raise RuntimeError("python-docx is not installed.")
    doc = docx.Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return text, bool(doc.inline_shapes)


def _extract_doc(path: Path) -> tuple[str, bool]:
    if textract is None:
        raise RuntimeError("textract is required for .doc files.")
    text = textract.process(str(path)).decode("utf-8", errors="ignore")
    return text, False


_EXTRACTORS = {
    "pdf":  _extract_pdf,
    "docx": _extract_docx,
    "doc":  _extract_doc,
}


def _fix_camel_case(text: str) -> str:
    """Split camelCase boundaries introduced by PDF extraction,
    but leave URLs, emails, and path-like tokens untouched."""
    def _replace(m: re.Match) -> str:
        token_start = m.start()
        while token_start > 0 and text[token_start - 1] not in (" ", "\n", "\t"):
            token_start -= 1
        token = text[token_start: m.end()]
        if any(c in token for c in (".", "/", "@", "\\", ":")):
            return m.group(0)
        return m.group(1) + " " + m.group(2)

    return re.sub(r"([a-z])([A-Z])", _replace, text)


def _normalise_tech(text: str) -> str:
    """Apply TECH_NORMALIZATION substitutions."""
    for pattern, replacement in TECH_NORMALIZATION.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── Public API ─────────────────────────────────────────────────────────────────
def extract_text_and_photo(path: Path) -> tuple[str, bool]:
    """Extract plain text and photo presence from a resume file."""
    if not path.exists():
        raise FileNotFoundError("Uploaded file is unavailable.")

    ext = path.suffix.lower().lstrip(".")
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError("Unsupported file format.")

    text, has_photo = extractor(path)
    text = _fix_camel_case(text)
    text = _normalise_tech(text)
    return text.strip(), has_photo
