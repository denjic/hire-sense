"""Curated job-description templates, 3 per resume category."""

from __future__ import annotations

from src.data.constants import CANONICAL_JOB_CATEGORIES

# Each entry is (seniority_or_focus, body). Bodies are written as real postings,
# not keyword dumps, so later Sentence-BERT training sees natural language.
JD_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Advocate": [
        (
            "Litigation Associate",
            "We are hiring a Litigation Associate to represent clients in civil and commercial matters. "
            "You will draft pleadings, affidavits, and legal notices; appear in district and high court hearings; "
            "and manage case files from intake through judgment. The role requires strong command of CPC, Evidence Act, "
            "and contract law, plus the ability to research precedents and write clear legal opinions. "
            "You will coordinate with clients, opposing counsel, and court staff, and keep matter trackers current. "
            "Ideal candidates have 2+ years of courtroom exposure, excellent written advocacy, and comfort with tight deadlines.",
        ),
        (
            "Corporate Counsel",
            "Our legal team needs a Corporate Counsel to support commercial contracting and corporate compliance. "
            "Responsibilities include reviewing vendor and customer agreements, flagging indemnities and liability caps, "
            "advising business units on employment and data-protection issues, and maintaining a clause playbook. "
            "You will partner with finance and operations on due diligence for partnerships and support board minutes. "
            "We look for someone who can translate legal risk into practical options, not just redlines. "
            "Experience with Companies Act filings and NDAs is required; litigation background is a plus.",
        ),
        (
            "Legal Advisor — Family & Property",
            "A growing practice is seeking a Legal Advisor focused on family, succession, and property disputes. "
            "You will counsel clients on partition, title verification, will drafting, and mediation before filing. "
            "Day-to-day work includes client interviews, drafting petitions, coordinating with local counsel, and "
            "preparing clients for testimony. Empathy and discretion are as important as procedural knowledge. "
            "Candidates should know stamp-duty rules, land records, and typical timelines in civil courts. "
            "This is a client-facing role with a mix of advisory work and active litigation support.",
        ),
    ],
    "Arts": [
        (
            "Visual Arts Instructor",
            "We are looking for a Visual Arts Instructor to design studio courses in drawing, painting, and mixed media. "
            "You will write lesson plans aligned to skill levels, critique student work, and run workshops on color theory "
            "and composition. The role includes preparing exhibitions of student work, managing studio inventory, and "
            "collaborating with performing-arts faculty on interdisciplinary shows. "
            "A portfolio demonstrating figurative and conceptual work is required. Teaching experience in schools or "
            "community programs is strongly preferred. You should be comfortable giving constructive feedback in group settings.",
        ),
        (
            "Gallery / Exhibition Coordinator",
            "A contemporary gallery seeks an Exhibition Coordinator to manage the full lifecycle of shows. "
            "You will liaise with artists on install plans, write wall text and catalog copy, schedule openings, and "
            "handle condition reports for incoming works. The job also covers social-media teasers, collector walkthroughs, "
            "and coordinating lighting and AV with technicians. Attention to detail and calm problem-solving during install "
            "weeks are essential. Experience with inventory databases and rights/reproduction paperwork is a plus.",
        ),
        (
            "Creative Producer — Arts Programs",
            "Join our cultural organization as a Creative Producer for public arts programs. "
            "You will commission artists, build production timelines and budgets, and deliver festivals, residencies, "
            "and community workshops. Success means balancing artistic vision with permits, vendors, and accessibility. "
            "We need someone who can write grant narratives, brief designers, and debrief after events with clear notes. "
            "Background in theater, visual arts, or cultural management is welcome. Strong stakeholder communication is required.",
        ),
    ],
    "Automation Testing": [
        (
            "SDET — Test Automation",
            "We need a Software Development Engineer in Test to own UI and API automation for a SaaS product. "
            "You will design a Selenium/Playwright and REST-assured (or pytest) framework, integrate suites into CI, "
            "and reduce flaky tests. Responsibilities include writing page objects, data-driven tests, and publishing "
            "failure reports with logs and screenshots. You will partner with developers on testability and shift-left reviews. "
            "Required: Java or Python, Git, and a track record of stable pipelines. Experience with Dockerized browsers and "
            "Allure or similar reporting is a plus.",
        ),
        (
            "QA Automation Engineer — CI/CD",
            "Our platform team is hiring a QA Automation Engineer to embed tests in GitHub Actions / Jenkins. "
            "You will convert high-value regression cases into automated checks, gate merges on smoke suites, and "
            "maintain environment fixtures. The role includes API contract tests, basic performance smoke tests, and "
            "mentoring manual testers on automation patterns. We expect clear defect reports with steps, expected vs actual, "
            "and traces. Familiarity with TestNG/JUnit, Maven/Gradle, and mocking services is important.",
        ),
        (
            "Mobile Automation Tester",
            "Seeking a Mobile Automation Tester for Android and iOS apps. You will build Appium (or Espresso/XCUITest) "
            "coverage for critical user journeys, run suites on device farms, and triage OS-specific failures. "
            "Work includes gesture-heavy flows, push-notification scenarios, and offline-mode checks. "
            "You will collaborate with product on release criteria and keep a living automation backlog. "
            "Experience with Charles/Proxyman, deep linking, and accessibility checks is valued. "
            "Strong SQL or log-query skills help when validating backend side effects.",
        ),
    ],
    "Blockchain": [
        (
            "Smart Contract Engineer",
            "We are hiring a Smart Contract Engineer to design, implement, and audit Solidity contracts on EVM chains. "
            "You will write upgradeable patterns, tests with Foundry or Hardhat, and gas-efficient storage layouts. "
            "The role covers threat modeling (reentrancy, oracle risk, access control), coordinating external audits, "
            "and documenting admin keys and pause procedures. You will also review frontend wallet integration assumptions. "
            "Required: production Solidity, OpenZeppelin familiarity, and comfort explaining risks to non-engineers. "
            "Experience with L2s and subgraph indexing is a plus.",
        ),
        (
            "Blockchain Backend Developer",
            "Join us as a Blockchain Backend Developer building indexers, relayers, and APIs around on-chain events. "
            "You will parse logs, maintain canonical chain state, and expose reliable REST/GraphQL for product teams. "
            "Responsibilities include handling reorgs, nonce management for transactions, and monitoring mempool/RPC health. "
            "Stack typically includes Node.js or Python, PostgreSQL, and Redis. We need someone who treats keys and "
            "hot wallets as production incidents waiting to happen. Prior DeFi, NFT, or payments-on-chain work is welcome.",
        ),
        (
            "Web3 Product Engineer",
            "A Web3 Product Engineer is needed to ship user-facing dApp features with secure wallet UX. "
            "You will integrate wagmi/ethers, design transaction states (pending, replaced, failed), and work with "
            "designers on signing prompts that users actually understand. You will also write contract ABIs into the app "
            "and add analytics around drop-off in connect/sign flows. Understanding of EIP-712, gas estimation, and "
            "common phishing patterns is required. This is a full-stack role with a security-first mindset.",
        ),
    ],
    "Business Analyst": [
        (
            "IT Business Analyst",
            "We are looking for an IT Business Analyst to turn stakeholder needs into clear requirements and user stories. "
            "You will run workshops, map as-is/to-be processes, write acceptance criteria, and maintain a prioritized backlog "
            "with product and engineering. The role includes impact analysis for change requests, UAT coordination, and "
            "traceability from requirement to test case. Strong SQL for data validation and comfort with Jira/Confluence are expected. "
            "You should be able to challenge vague asks and document decisions, not just take notes.",
        ),
        (
            "Business Analyst — Finance Domain",
            "A finance-domain Business Analyst is needed for reporting, reconciliations, and regulatory change. "
            "You will elicit rules from controllers, document data lineage, and specify exception workflows. "
            "Day-to-day work includes BRD/FRD writing, mock reports, and sitting with QA on scenario coverage. "
            "Familiarity with GL, AP/AR, or risk metrics is important. Experience with Excel models plus a BI tool "
            "(Power BI/Tableau) helps you validate numbers before they hit production. Calm communication under month-end pressure is a must.",
        ),
        (
            "Agile Business Analyst",
            "Join a product squad as an Agile Business Analyst. You will refine stories, split epics, and keep the definition "
            "of done honest. Responsibilities include customer interviews, journey maps, and lightweight process diagrams. "
            "You will demo with the team, capture feedback, and update the backlog the same day. "
            "We value BAs who can quantify outcomes (conversion, cycle time, defect escape) rather than only listing features. "
            "Prior work in SaaS or internal tools is preferred. Facilitation skills matter as much as documentation.",
        ),
    ],
    "Civil Engineer": [
        (
            "Site Civil Engineer",
            "We need a Site Civil Engineer to supervise earthworks, drainage, and structural execution on active sites. "
            "You will interpret drawings and BOQs, check formwork and rebar before pours, and enforce safety and quality checklists. "
            "The role includes daily progress reports, coordinating subcontractors, and raising RFIs when drawings conflict with ground conditions. "
            "Knowledge of IS codes, setting-out, and concrete mix control is required. You should be comfortable with AutoCAD "
            "markups and talking to clients during inspections. 2+ years of site experience is expected.",
        ),
        (
            "Structural Design Engineer",
            "A consultancy is hiring a Structural Design Engineer for RCC and steel buildings. "
            "You will model loads, design members, produce GA and detail drawings, and review fabricator shop drawings. "
            "Software typically includes STAAD/ETABS and AutoCAD or Revit. You will write design basis notes and "
            "respond to peer-review comments. Attention to serviceability, seismic detailing, and constructability is essential. "
            "We look for engineers who document assumptions and flag when architecture changes break the structural scheme.",
        ),
        (
            "Quantity Survey / Estimation Engineer",
            "Seeking an Estimation Engineer to prepare tenders and control quantities during execution. "
            "You will take off quantities from drawings, rate analysis, compare vendor quotes, and track variations. "
            "The job includes BOQ updates, cash-flow inputs, and claims support with measurement sheets. "
            "Accuracy and version control of estimates are critical. Experience with residential or infrastructure projects "
            "and familiarity with CPWD/state SOR is an advantage. You will work closely with planning and procurement.",
        ),
    ],
    "Data Science": [
        (
            "Data Scientist — Applied ML",
            "We are hiring a Data Scientist to build and evaluate predictive models for a product team. "
            "You will frame problems, engineer features, train models in Python (pandas, scikit-learn, optionally PyTorch), "
            "and communicate lift versus a simple baseline. Responsibilities include experiment design, leakage checks, "
            "and documenting why a model should or should not ship. You will partner with engineers on inference constraints. "
            "SQL fluency and clear writing are required. Experience with classification, ranking, or forecasting in production is a plus.",
        ),
        (
            "ML Engineer / Data Scientist",
            "Join us as an ML-leaning Data Scientist who can take notebooks to scheduled jobs. "
            "You will own data quality checks, training pipelines, and monitoring for drift and performance. "
            "The role includes feature stores or well-named tables, model cards, and rollback plans. "
            "Stack: Python, SQL, Git, and a cloud object store. We care about reproducibility and honest metrics, "
            "not leaderboard chasing. Prior work with NLP, recommenders, or tabular business models is welcome.",
        ),
        (
            "Analytics Data Scientist",
            "An Analytics Data Scientist is needed to turn messy business questions into trustworthy analyses. "
            "You will define metrics, build cohorts, run A/B readouts, and present findings to non-technical leads. "
            "Day-to-day includes SQL, Python visualization, and stakeholder workshops to separate correlation from action. "
            "You should be skeptical of vanity metrics and explicit about confidence intervals and sample size. "
            "Experience in product analytics, marketing measurement, or operations research is a strong fit.",
        ),
    ],
    "Database": [
        (
            "Database Administrator",
            "We need a Database Administrator to keep PostgreSQL/MySQL (or SQL Server) healthy in production. "
            "You will manage backups and restores, replication, parameter tuning, and on-call for lock/blocking incidents. "
            "Responsibilities include capacity planning, index reviews, and working with developers on slow-query plans. "
            "Security work covers roles, least privilege, and encryption at rest/in transit. "
            "Experience with point-in-time recovery and documenting runbooks is required. Cloud managed databases are a plus.",
        ),
        (
            "SQL Developer",
            "Hiring a SQL Developer to design schemas, stored procedures, and reporting views. "
            "You will write set-based queries, enforce constraints, and partner with analysts on dimensional models. "
            "The role includes code reviews for SQL, migration scripts, and performance tests on realistic data volumes. "
            "We expect naming conventions, comments on complex logic, and awareness of transaction isolation. "
            "ETL collaboration and basic Python/PowerShell for ops tasks are helpful but not the core of the job.",
        ),
        (
            "Data Platform Engineer — Databases",
            "A Data Platform Engineer focused on databases will own the path from OLTP to analytical stores. "
            "You will design CDC or batch extract patterns, manage connection pooling, and set SLAs for replica lag. "
            "Work includes partitioning strategies, vacuum/maintenance windows, and cost control on cloud IOPS. "
            "You will be the escalation point when an application change threatens the database. "
            "PostgreSQL expertise is preferred; Oracle or SQL Server experience is transferable.",
        ),
    ],
    "DevOps Engineer": [
        (
            "DevOps Engineer — CI/CD & Cloud",
            "We are hiring a DevOps Engineer to own CI/CD, infrastructure as code, and production reliability. "
            "You will build pipelines (GitHub Actions/Jenkins), containerize services, and manage Kubernetes or ECS. "
            "Responsibilities include secrets management, observability (metrics, logs, traces), and incident response. "
            "Terraform or equivalent IaC is expected. You should be able to explain blast radius of a change and "
            "put guardrails around deploys. Linux fluency and a bias toward automation over tribal knowledge are required.",
        ),
        (
            "SRE-leaning DevOps",
            "Join as a DevOps/SRE hybrid to raise reliability of customer-facing APIs. "
            "You will define SLIs/SLOs, implement alerting that pages on symptoms not noise, and run post-incident reviews. "
            "The role includes capacity tests, chaos-lite drills, and working with developers on graceful degradation. "
            "Experience with Prometheus/Grafana or cloud-native equivalents is important. "
            "We need someone who treats toil as a backlog item, not a personality trait.",
        ),
        (
            "Platform DevOps Engineer",
            "A Platform DevOps Engineer is needed to provide paved roads for product teams: golden pipelines, "
            "base images, and environment provisioning. You will reduce ticket-driven ops by self-service docs and templates. "
            "Work includes IAM reviews, network policies, and cost dashboards. Strong written documentation is part of the job. "
            "Prior experience supporting multiple teams from a central platform is a plus. Python or Go for internal tools helps.",
        ),
    ],
    "DotNet Developer": [
        (
            "ASP.NET Core Developer",
            "We need an ASP.NET Core Developer to build and maintain APIs and internal web apps. "
            "You will write C# services, Entity Framework mappings, and authentication/authorization with identity patterns. "
            "Responsibilities include unit/integration tests, code reviews, and performance work on hot endpoints. "
            "Familiarity with SQL Server, REST design, and Azure or on-prem IIS deployments is expected. "
            "We look for developers who handle nullability, logging, and configuration via environment — not hardcoded secrets.",
        ),
        (
            "Full-stack .NET Engineer",
            "Hiring a full-stack .NET Engineer comfortable with C# backends and a modern SPA or Razor/Blazor frontend. "
            "You will ship features end-to-end: data model, API, UI, and telemetry. The role includes accessibility basics "
            "and collaborating with QA on regression risk. Experience with message queues or background workers is welcome. "
            "Clean architecture is a guideline, not a religion — we care about readable modules and tested boundaries.",
        ),
        (
            "Legacy .NET Modernization Developer",
            "A modernization team needs a .NET Developer to migrate Framework apps toward .NET 8+ and cleaner APIs. "
            "You will inventory dependencies, replace brittle COM/WCF pieces, and add tests before refactors. "
            "The job includes pairing with business users to confirm behavior, not just compiling. "
            "Patience with legacy code and a methodical strangler-fig approach are more valuable than greenfield-only experience.",
        ),
    ],
    "ETL Developer": [
        (
            "ETL Developer — Informatica / SSIS",
            "We are hiring an ETL Developer to design batch pipelines from operational systems into a warehouse. "
            "You will build mappings, handle slowly changing dimensions, and implement error/reject handling with audit tables. "
            "Responsibilities include job scheduling, restartability, and data-quality checks (uniqueness, referential integrity). "
            "Experience with Informatica, SSIS, or equivalent is required; SQL is non-negotiable. "
            "You should document source-to-target mappings so the next person can debug at 2 a.m.",
        ),
        (
            "Data Engineer — ETL/ELT",
            "Join as a Data Engineer focused on ELT in a cloud warehouse (Snowflake/BigQuery/Redshift). "
            "You will write dbt or SQL transformations, manage incremental models, and set freshness tests. "
            "The role includes orchestrating with Airflow or similar and partnering with analysts on mart design. "
            "We care about idempotent jobs and explicit grain. Python for ingestion connectors is a plus. "
            "Prior experience with CDC or API extracts is welcome.",
        ),
        (
            "ETL Developer — Healthcare / Finance feeds",
            "Seeking an ETL Developer for high-stakes feeds where late or wrong data has real cost. "
            "You will implement reconciliation reports, late-arriving fact handling, and PII handling rules. "
            "Work includes vendor file specs, schema drift detection, and SLAs with downstream reporting. "
            "Calm communication when a source file is late or malformed is part of the job. "
            "Strong SQL, file formats (CSV, XML, JSON), and logging discipline are required.",
        ),
    ],
    "Electrical Engineering": [
        (
            "Electrical Design Engineer",
            "We need an Electrical Design Engineer for power distribution, lighting, and control schematics. "
            "You will prepare SLDs, cable schedules, and protection calculations, then review contractor shop drawings. "
            "The role includes coordination with mechanical and civil disciplines and site queries during installation. "
            "Knowledge of IS/IEC standards, earthing, and short-circuit studies is expected. AutoCAD Electrical or similar "
            "is used daily. Safety and maintainability of designs matter as much as meeting load lists.",
        ),
        (
            "Field Electrical Engineer",
            "Hiring a Field Electrical Engineer to supervise installation, testing, and commissioning of electrical systems. "
            "You will witness IR/PI tests, check panel termination quality, and close punch lists. "
            "The job includes permit-to-work discipline, lockout/tagout, and coordinating outages with operations. "
            "Hands-on experience with MCCs, VFDs, and transformers is valued. Clear daily reporting to the project manager is required.",
        ),
        (
            "Embedded / Controls Electrical Engineer",
            "A product company seeks an Electrical Engineer for embedded controls and industrial automation. "
            "You will work on PLC/HMI logic or microcontroller-adjacent hardware bring-up, sensor interfacing, and EMC-aware layout reviews. "
            "Responsibilities include test plans, failure analysis, and working with firmware on timing and I/O maps. "
            "Experience with motors, power electronics, or building-management systems is a plus. "
            "We need someone who documents wiring changes instead of leaving tribal knowledge in the panel.",
        ),
    ],
    "Hadoop": [
        (
            "Hadoop Developer",
            "We are hiring a Hadoop Developer to build batch processing on HDFS with Spark/Hive. "
            "You will write Spark jobs (PySpark or Scala), design partitioned tables, and tune shuffles and file sizes. "
            "Responsibilities include ingest from Kafka or Sqoop-like patterns, and making jobs recoverable. "
            "You will work with platform teams on YARN/K8s resource settings and with analysts on Hive metastore tables. "
            "Understanding of small-files problems, data skew, and exactly-once vs at-least-once is expected.",
        ),
        (
            "Big Data Engineer — Hadoop Stack",
            "Join as a Big Data Engineer owning the Hadoop ecosystem pieces still in production: Hive, HBase or Kudu, and Oozie/Airflow. "
            "You will migrate brittle jobs toward Spark SQL, add lineage notes, and reduce cluster cost. "
            "The role includes on-call for failed jobs and capacity conversations with the platform team. "
            "Linux, Kerberos basics, and SQL at scale are required. Experience with Cloudera or similar distributions is helpful.",
        ),
        (
            "Spark / Hadoop Data Engineer",
            "A Spark-first Data Engineer is needed to process large event and log datasets. "
            "You will design streaming or micro-batch pipelines, enforce schemas, and publish curated datasets. "
            "Work includes testing with sampled data, monitoring job duration, and documenting SLAs. "
            "Python or Scala plus strong SQL is the core toolkit. Hadoop HDFS experience is required even if the future is object storage.",
        ),
    ],
    "Health and fitness": [
        (
            "Fitness Coach / Personal Trainer",
            "We are looking for a Fitness Coach to design programs for members with varied goals and injury histories. "
            "You will run assessments, write periodized plans, cue movement safely, and track progress without ego. "
            "The role includes small-group classes, nutrition-habit coaching within scope of practice, and floor supervision. "
            "Certification (ACE, NASM, or equivalent) and CPR/AED are required. "
            "We need coaches who escalate medical red flags instead of playing doctor, and who keep session notes professional.",
        ),
        (
            "Wellness Program Coordinator",
            "A corporate wellness team needs a Program Coordinator to run fitness challenges, workshops, and vendor classes. "
            "You will schedule instructors, measure participation, and report outcomes to HR without sharing individual health details. "
            "Responsibilities include venue logistics, communications, and adapting programs for beginners and experienced athletes. "
            "Experience in gyms, physiotherapy admin, or public health outreach is welcome. Strong organization is the main skill.",
        ),
        (
            "Strength & Conditioning Coach",
            "Hiring a Strength & Conditioning Coach for athletic or serious recreational populations. "
            "You will program strength, power, and energy-system work, manage load, and coordinate with physios on return-to-play. "
            "The job includes testing (jump, strength, aerobic), video review of lifts, and educating athletes on recovery. "
            "A CSCS or comparable credential is preferred. We value coaches who can explain why a progression exists.",
        ),
    ],
    "HR": [
        (
            "HR Generalist",
            "We need an HR Generalist to own the employee lifecycle: recruiting coordination, onboarding, policy questions, and exits. "
            "You will maintain employee records, support performance-cycle logistics, and advise managers on basic labor-law hygiene. "
            "The role includes running engagement pulses, coordinating training calendars, and handling sensitive conversations with discretion. "
            "HRIS experience and clear writing are required. You should know when to escalate to legal or a specialist, not bluff.",
        ),
        (
            "Talent Acquisition Specialist",
            "Hiring a Talent Acquisition Specialist to fill technical and business roles with a structured process. "
            "You will write scorecards with hiring managers, source candidates, run first-round screens, and keep a fair pipeline. "
            "Responsibilities include scheduling, closing, and reporting funnel metrics (time-to-fill, source quality). "
            "We expect evidence-based interviewing, not vibe-based hiring. Experience with LinkedIn Recruiter or equivalent is a plus.",
        ),
        (
            "HR Operations / Payroll Support",
            "An HR Operations role focused on payroll inputs, benefits admin, and compliance calendars. "
            "You will audit attendance data, process changes, and answer employee questions with documented SOPs. "
            "Accuracy and confidentiality are the job. Experience with statutory filings (as applicable in your region) is valued. "
            "You will partner with finance on reconciliations and with IT on access-provisioning checklists for joiners/leavers.",
        ),
    ],
    "Java Developer": [
        (
            "Java Backend Developer",
            "We are hiring a Java Backend Developer to build Spring Boot services. "
            "You will design REST APIs, persist with JPA/Hibernate, and write tests that fail for the right reasons. "
            "Responsibilities include code reviews, logging/metrics on endpoints, and handling transactions and idempotency. "
            "Experience with Maven/Gradle, Git, and SQL is required. Kafka or similar messaging is a plus. "
            "We look for developers who understand concurrency pitfalls and don't swallow exceptions.",
        ),
        (
            "Java Microservices Engineer",
            "Join a microservices team writing Java services with clear bounded contexts. "
            "You will implement resilience (timeouts, retries with jitter, bulkheads), contract tests, and OpenAPI specs. "
            "The role includes on-call for your services and improving local developer experience. "
            "Docker familiarity and CI pipelines are expected. Prior experience with distributed tracing helps debug real incidents.",
        ),
        (
            "Senior Java Developer — Enterprise",
            "A Senior Java Developer is needed for a large enterprise domain (payments, insurance, or similar). "
            "You will mentor juniors, break down designs, and keep legacy modules from rotting while new features ship. "
            "Work includes performance tuning, batch jobs, and careful schema migrations. "
            "Strong communication with BA and QA partners is required. We value engineers who leave the codebase clearer than they found it.",
        ),
    ],
    "Mechanical Engineer": [
        (
            "Mechanical Design Engineer",
            "We need a Mechanical Design Engineer to own 3D models, drawings, and GD&T for manufactured parts. "
            "You will work in SolidWorks/CATIA/Creo, run basic FEA or hand calcs, and release drawings through a PDM system. "
            "Responsibilities include DFM reviews with vendors, ECO/ECN handling, and prototype bring-up. "
            "Understanding of tolerances, materials, and manufacturing processes (CNC, sheet metal, molding) is required. "
            "You should be able to explain a design choice in a design review without hiding behind the CAD.",
        ),
        (
            "Production / Manufacturing Engineer",
            "Hiring a Manufacturing Engineer to improve yield, cycle time, and safety on the shop floor. "
            "You will write work instructions, design fixtures, and lead root-cause on defects (5-Why, Pareto). "
            "The role includes line balancing, tooling follow-up, and coordinating with quality on control plans. "
            "Experience in automotive, industrial equipment, or similar discrete manufacturing is a plus. "
            "Hands-on credibility with operators matters more than slide decks.",
        ),
        (
            "HVAC / Thermal Mechanical Engineer",
            "A Thermal/HVAC Mechanical Engineer is needed for load calculations, duct/pipe layouts, and equipment selection. "
            "You will produce schematics, coordinate with electrical and architecture, and support commissioning. "
            "Knowledge of psychrometrics, pumps, and energy codes is expected. Site visits and vendor submittal reviews are part of the job. "
            "We look for engineers who size systems for maintainability, not just first-cost.",
        ),
    ],
    "Network Security Engineer": [
        (
            "Network Security Engineer",
            "We are hiring a Network Security Engineer to design and operate perimeter and internal controls. "
            "You will manage firewalls, VPNs, segmentation, and IDS/IPS tuning, plus review change requests for blast radius. "
            "Responsibilities include packet analysis during incidents, hardening network devices, and documenting allow-lists. "
            "Experience with Cisco/Palo Alto/Fortinet or equivalent and a solid TCP/IP mental model are required. "
            "You should be able to say no with an alternative, not just a ticket rejection.",
        ),
        (
            "SOC / Network Security Analyst (Engineer track)",
            "Join as a security engineer supporting SOC detections on network telemetry. "
            "You will write detections, reduce false positives, and hunt using flow logs and DNS data. "
            "The role includes tabletop exercises, threat-intel application, and working with IT on containment. "
            "SIEM experience (Splunk, Sentinel, or similar) is expected. Clear incident notes are a deliverable, not an afterthought.",
        ),
        (
            "Zero Trust / Identity-aware Network Engineer",
            "Seeking a Network Security Engineer to implement identity-aware access and reduce flat networks. "
            "You will design micro-segmentation, certificate-based access, and secure remote work patterns. "
            "Work includes PKI hygiene, NAC, and partnering with IAM on least privilege. "
            "Cloud security groups and on-prem ACLs both appear in this role. Documentation of trust boundaries is required.",
        ),
    ],
    "Operations Manager": [
        (
            "Operations Manager — Service Delivery",
            "We need an Operations Manager to own daily service delivery, SLAs, and cross-team coordination. "
            "You will run stand-ups that surface blockers, manage vendor performance, and keep a living ops dashboard. "
            "Responsibilities include capacity planning, incident command for major outages, and after-action improvements. "
            "Experience leading coordinators or team leads is required. We value managers who inspect the work, not only the slides.",
        ),
        (
            "Plant / Facility Operations Manager",
            "Hiring an Operations Manager for a plant or multi-site facility. "
            "You will own safety metrics, maintenance schedules, inventory of critical spares, and shift handover quality. "
            "The role includes budgeting, contractor management, and compliance audits. "
            "Calm decision-making when equipment fails and people are waiting is essential. "
            "Background in manufacturing, logistics, or facilities is expected.",
        ),
        (
            "Business Operations Manager",
            "A Business Operations Manager is needed to make processes measurable and less heroic. "
            "You will map workflows, remove handoff delays, and implement simple trackers that teams actually use. "
            "Work includes OKR mechanics, vendor onboarding, and partnering with finance on unit economics. "
            "Strong written communication and a bias to document decisions are required. SQL or spreadsheet modeling is a plus.",
        ),
    ],
    "PMO": [
        (
            "PMO Analyst",
            "We are hiring a PMO Analyst to keep project status honest and comparable across the portfolio. "
            "You will collect RAID logs, chase milestone evidence, and produce steering-committee packs that highlight risk, not color. "
            "Responsibilities include maintaining the project register, dependency maps, and change-control records. "
            "Comfort with MS Project/Jira/Smartsheet and executive-ready writing is required. "
            "You should challenge green status that has no underlying metrics.",
        ),
        (
            "Project Manager (PMO)",
            "Join the PMO as a Project Manager delivering scoped initiatives on time and within governance. "
            "You will build plans, run RAID, manage stakeholders, and close projects with lessons learned that get reused. "
            "The role includes budget tracking, vendor coordination, and UAT planning with business owners. "
            "PMP/PRINCE2 is nice; delivery evidence is nicer. Facilitation and conflict management are core skills.",
        ),
        (
            "PMO Lead — Governance",
            "A PMO Lead is needed to define stage gates, templates, and portfolio reviews without drowning teams in paperwork. "
            "You will coach PMs, escalate systemic resource conflicts, and report investment vs outcome to leadership. "
            "Experience implementing a PMO or recovering a chaotic portfolio is valued. "
            "We need someone who can distinguish process that reduces risk from process that only produces decks.",
        ),
    ],
    "Python Developer": [
        (
            "Python Backend Developer",
            "We need a Python Backend Developer to build FastAPI or Django services. "
            "You will design APIs, write pytest coverage, and integrate with PostgreSQL and Redis. "
            "Responsibilities include typing, structured logging, and configuration via environment variables. "
            "Experience with Celery/RQ or similar workers is a plus. We look for developers who handle errors and timeouts explicitly. "
            "Code reviews and mentoring juniors on Pythonic style are part of the culture.",
        ),
        (
            "Python Engineer — Data/Platform",
            "Hiring a Python Engineer to build internal data and platform tools. "
            "You will write CLI utilities, batch jobs, and small services that other teams depend on. "
            "The role includes packaging, CI, and making failure modes obvious. Pandas/SQL appears often; this is still an engineering job. "
            "Experience with cloud SDKs and object storage is welcome. Reliability beats clever one-liners.",
        ),
        (
            "Full-stack Python Developer",
            "A full-stack Python Developer is needed for a product that uses Python on the server and a lightweight frontend. "
            "You will ship features across API and UI, own migrations, and watch production logs after deploy. "
            "Familiarity with auth, file uploads, and background tasks is expected. "
            "We value product sense and the ability to cut scope without cutting correctness.",
        ),
    ],
    "SAP Developer": [
        (
            "SAP ABAP Developer",
            "We are hiring an SAP ABAP Developer to deliver RICEF objects with clean transport discipline. "
            "You will write reports, enhancements, forms, and interfaces, and debug production issues with ST22/SM37 evidence. "
            "Responsibilities include performance (SQL in ABAP), unit testing where available, and documenting functional specs vs technical specs. "
            "Experience with S/4HANA or ECC is required; OData/Fiori exposure is a plus. "
            "You should refuse silent implicit enhancements that nobody can find later.",
        ),
        (
            "SAP Functional-Technical Consultant",
            "Join as a techno-functional SAP consultant who can sit with business and still ship configuration/code. "
            "You will gather requirements in MM/SD/FI (or your module), propose standard vs custom, and test end-to-end processes. "
            "The role includes cutover support and hypercare. Strong workshop facilitation and written functional specs are required. "
            "Integration knowledge (IDoc, PI/PO, CPI) is valued.",
        ),
        (
            "SAP Integration Developer",
            "Seeking an SAP Integration Developer for APIs, IDocs, and middleware mappings. "
            "You will design interfaces, handle retries and monitoring, and partner with non-SAP systems owners. "
            "Experience with SAP CPI, PI/PO, or event-based patterns is expected. "
            "Data quality and idempotent processing matter as much as the mapping spreadsheet. "
            "On-call for interface failures is part of the role during go-lives.",
        ),
    ],
    "Sales": [
        (
            "B2B Sales Executive",
            "We need a B2B Sales Executive to own a territory or segment from prospecting to close. "
            "You will run discovery, map stakeholders, forecast honestly, and write proposals that match the problem, not a feature dump. "
            "Responsibilities include CRM hygiene, pipeline reviews, and coordinating with solutions/pre-sales. "
            "Experience selling to mid-market or enterprise is required. We measure activity quality and win-loss learning, not just vanity calls.",
        ),
        (
            "Inside Sales / SDR",
            "Hiring an SDR/Inside Sales rep to generate qualified meetings for account executives. "
            "You will research accounts, personalize outreach, and qualify on budget, authority, need, and timing without being robotic. "
            "The role includes A/B testing messaging and logging outcomes in the CRM the same day. "
            "Resilience and coachability matter more than a perfect script. Prior SaaS or services sales is a plus.",
        ),
        (
            "Account Manager — Retention & Expansion",
            "An Account Manager is needed to retain and expand existing customers. "
            "You will run QBRs, spot adoption gaps, and negotiate renewals with a clear value narrative. "
            "Work includes coordinating support and product on at-risk accounts. "
            "Empathy plus commercial backbone is the combination we hire for. Experience with churn analysis is welcome.",
        ),
    ],
    "Testing": [
        (
            "Manual QA Engineer",
            "We are hiring a Manual QA Engineer to design and execute test cases for web and API products. "
            "You will write clear test cases, exploratory charters, and defect reports with environment, data, and logs. "
            "Responsibilities include regression planning, UAT support, and risk-based coverage when time is short. "
            "You should challenge requirements that cannot be tested. SQL for data setup and browser/dev-tools fluency are expected. "
            "ISTQB is optional; evidence of finding high-severity bugs is not.",
        ),
        (
            "QA Engineer — Functional & Regression",
            "Join as a QA Engineer owning functional coverage for a release train. "
            "You will maintain a living regression suite, triage failures, and partner with developers on root cause. "
            "The role includes test-data strategy and environment sanity checks before a cycle starts. "
            "Experience with Jira/TestRail (or equivalent) and API tools like Postman is required. "
            "We value testers who think in user journeys and failure modes, not only happy paths.",
        ),
        (
            "Quality Analyst — Domain Testing",
            "A domain-focused Quality Analyst is needed (finance, healthcare, or similar regulated flows). "
            "You will build scenario matrices from business rules, check calculations, and verify audit trails. "
            "Work includes negative testing, boundary values, and coordinating with BA on ambiguous rules. "
            "Patience with documentation and a skeptical eye for silent data loss are essential. "
            "Prior domain experience is a strong advantage.",
        ),
    ],
    "Web Designing": [
        (
            "UI Designer / Web Designer",
            "We need a Web Designer to produce responsive marketing and product pages that are usable, not just decorative. "
            "You will create layouts in Figma, specify typography and spacing, and hand off assets with redlines. "
            "Responsibilities include accessibility (contrast, focus states), component consistency, and collaborating with developers on feasibility. "
            "A strong portfolio of live sites is required. Motion should support hierarchy, not distract. "
            "Experience with design systems is a plus.",
        ),
        (
            "Frontend-leaning Web Designer",
            "Hiring a Web Designer who can implement in HTML/CSS (and light JS) as well as design. "
            "You will build landing pages, fix cross-browser issues, and keep a component library tidy. "
            "The role includes performance basics (image sizes, font loading) and working with content authors. "
            "We look for people who test on real phones. Familiarity with Webflow or a static-site workflow is welcome.",
        ),
        (
            "UX/UI Designer — Web Apps",
            "A UX/UI Designer is needed for a web application: flows, empty states, and error states included. "
            "You will run lightweight research, prototype, and iterate with product and engineering. "
            "Deliverables include user flows, high-fidelity UI, and usage of a shared component library. "
            "We care about reducing cognitive load and making destructive actions hard to do by accident. "
            "Prior SaaS or dashboard work is a strong fit.",
        ),
    ],
}


_ADDITIONAL_JD_FOCUSES: dict[str, tuple[str, ...]] = {
    "Advocate": ("Legal Research Counsel", "Compliance Counsel", "Dispute Resolution Specialist", "Employment Law Advisor", "Intellectual Property Counsel", "Legal Operations Analyst"),
    "Arts": ("Community Arts Educator", "Arts Operations Manager", "Collections Assistant", "Public Programs Curator", "Arts Marketing Coordinator", "Studio Manager"),
    "Automation Testing": ("API Test Engineer", "Performance Test Engineer", "Test Framework Developer", "Accessibility Test Engineer", "Security Test Engineer", "Quality Engineering Lead"),
    "Blockchain": ("Protocol Integration Engineer", "Blockchain Security Analyst", "Web3 QA Engineer", "Smart Contract Auditor", "Token Economics Analyst", "Blockchain Infrastructure Engineer"),
    "Business Analyst": ("Data Business Analyst", "Product Business Analyst", "Process Improvement Analyst", "Requirements Analyst", "Business Systems Analyst", "Customer Insights Analyst"),
    "Civil Engineer": ("Construction Planning Engineer", "QA/QC Civil Engineer", "Infrastructure Project Engineer", "Geotechnical Project Engineer", "Water Resources Engineer", "Site Safety Engineer"),
    "Data Science": ("NLP Data Scientist", "Decision Science Analyst", "Data Science Platform Engineer", "Computer Vision Scientist", "Data Science Consultant", "Machine Learning Researcher"),
    "Database": ("Database Reliability Engineer", "Database Migration Engineer", "Cloud Database Administrator", "Database Performance Engineer", "Data Modeling Engineer", "Database Security Engineer"),
    "DevOps Engineer": ("Cloud Infrastructure Engineer", "Release Engineer", "DevSecOps Engineer", "Kubernetes Platform Engineer", "Observability Engineer", "Infrastructure Automation Engineer"),
    "DotNet Developer": (".NET API Engineer", ".NET Integration Developer", ".NET Test Engineer", ".NET Cloud Developer", ".NET Desktop Developer", ".NET Solutions Architect"),
    "ETL Developer": ("ETL Quality Engineer", "Data Warehouse Developer", "Pipeline Reliability Engineer", "Data Integration Consultant", "ETL Performance Engineer", "Master Data Engineer"),
    "Electrical Engineering": ("Power Systems Engineer", "Electrical Commissioning Engineer", "Control Systems Engineer", "Protection Engineer", "Renewable Energy Engineer", "Electrical Maintenance Engineer"),
    "Hadoop": ("Hadoop Operations Engineer", "Data Lake Engineer", "Streaming Data Engineer", "Spark Applications Engineer", "Big Data Solutions Architect", "Data Infrastructure Analyst"),
    "Health and fitness": ("Health Coach", "Fitness Studio Manager", "Exercise Program Specialist", "Sports Rehabilitation Coach", "Group Exercise Instructor", "Wellness Content Specialist"),
    "HR": ("People Operations Specialist", "Learning and Development Coordinator", "HR Compliance Analyst", "Compensation and Benefits Analyst", "Employee Relations Partner", "HRIS Administrator"),
    "Java Developer": ("Java API Engineer", "Spring Boot Developer", "Java Application Support Engineer", "Java Performance Engineer", "Java Integration Specialist", "Java Technical Lead"),
    "Mechanical Engineer": ("Mechanical Project Engineer", "Design Verification Engineer", "Reliability Mechanical Engineer", "Product Development Engineer", "Industrial Automation Engineer", "Mechanical Quality Engineer"),
    "Network Security Engineer": ("Firewall Security Engineer", "Network Detection Engineer", "Cloud Network Security Engineer", "Incident Response Engineer", "Security Architecture Engineer", "Network Access Control Engineer"),
    "Operations Manager": ("Operations Excellence Manager", "Supply Chain Operations Manager", "Customer Operations Manager", "Service Operations Lead", "Workforce Operations Manager", "Process Operations Analyst"),
    "PMO": ("Portfolio Analyst", "Delivery Governance Manager", "Project Controls Analyst", "Transformation PMO Analyst", "Risk and Controls Manager", "PMO Reporting Specialist"),
    "Python Developer": ("Python API Engineer", "Python Automation Engineer", "Python Data Services Developer", "Django Application Engineer", "Python Cloud Engineer", "Python Test Engineer"),
    "SAP Developer": ("SAP Fiori Developer", "SAP ABAP Support Engineer", "SAP Data Migration Consultant", "SAP Basis Developer", "SAP Reporting Developer", "SAP Workflow Consultant"),
    "Sales": ("Solutions Sales Executive", "Sales Operations Analyst", "Customer Growth Manager", "Enterprise Account Executive", "Channel Sales Manager", "Sales Enablement Specialist"),
    "Testing": ("API QA Engineer", "Test Environment Analyst", "Release Quality Engineer", "Performance QA Engineer", "Security QA Analyst", "Test Lead"),
    "Web Designing": ("Design Systems Specialist", "Web Accessibility Designer", "Content and UX Designer", "Interaction Designer", "Visual Designer", "Web Production Specialist"),
}


for _category, _focuses in _ADDITIONAL_JD_FOCUSES.items():
    _base_body = JD_TEMPLATES[_category][0][1]
    for _focus in _focuses:
        JD_TEMPLATES[_category].append(
            (
                _focus,
                f"This {_category} role focuses on {_focus.lower()}. "
                "You will plan work, collaborate with stakeholders, document decisions, "
                "and deliver reliable outcomes against clear quality and timeline expectations. "
                f"The role builds on {_base_body.lower()}",
            )
        )


def iter_job_descriptions() -> list[dict[str, str]]:
    """Flatten templates into records with stable jd_id values."""
    records: list[dict[str, str]] = []
    missing = [cat for cat in CANONICAL_JOB_CATEGORIES if cat not in JD_TEMPLATES]
    if missing:
        raise RuntimeError(f"Missing JD templates for categories: {missing}")

    seq = 1
    for category in CANONICAL_JOB_CATEGORIES:
        for focus, body in JD_TEMPLATES[category]:
            text = " ".join(body.split())
            records.append(
                {
                    "jd_id": f"JD{seq:04d}",
                    "category": category,
                    "title": f"{category} — {focus}",
                    "jd_text": f"{category} — {focus}. {text}",
                }
            )
            seq += 1
    return records
