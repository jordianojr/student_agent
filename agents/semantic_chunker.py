import json
import re
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
import logging
import os
from dotenv import load_dotenv
import ollama
load_dotenv()

class SemanticChunker:
    def __init__(self):
        self.n_ctx = 10000


    def write_chunking_plan(self, content):
        '''
            Prompt to extract content ideas.

            Parameters:
            - content (String): Scraped and cleaned content

            Returns:
            - list: List of ideas found in the content
            '''
        # gemma-2-9b-it prompt
        summarize_template = '''
        <start_of_turn>user
        You are a diligent STEM student reviewing study notes for finals. Extract ALL important information as separate ideas.

        Extract from these slides:
        **FORMULAS & EQUATIONS:**
        - List every formula with variable definitions
        - Note when each formula applies (conditions/constraints)
        - Include units where relevant

        **KEY CONCEPTS & DEFINITIONS:**
        - Technical terms that will be tested
        - Theorems and their conditions
        - Physical/mathematical principles
        - Constants and their values

        **PROCEDURES & ALGORITHMS:**
        - Step-by-step methods
        - Decision trees for choosing approaches
        - Computational procedures

        **GENERAL IDEAS:**
        - Any other important points for exam prep

        <original_content>
        {content}
        </original_content>

        IMPORTANT: Format EVERY item as "Idea X:" followed by the BIG PICTURE idea. Number them sequentially.
        Break down complex ideas into multiple items if needed.
        Do not limit yourself to just 10 ideas; extract ALL relevant information.

        Example:
        Idea 1: Newton's second law: F = ma, where F is force (N), m is mass (kg), a is acceleration (m/s²)
        Idea 2: Kinematic equations apply only when acceleration is constant
        Idea 3: To solve projectile motion: separate x and y components, use appropriate kinematic equations for each

        <end_of_turn>
        <start_of_turn>model
        '''

        response = ollama.chat(
        model='qwen3:4b',
        messages=[
            {"role": "user", 
            "content": summarize_template.format(content=content)
            }
        ]
    )
        all_ideas = response['message']['content']
        print(f'Chunking plan created: \n{all_ideas}\n')

        pattern = r"Idea \d+: (.*)"
        matches = re.findall(pattern, all_ideas)
        list_of_ideas = [idea.strip() for idea in matches]

        return list_of_ideas

    def extract_chunk(self, idea, text):
        extract_template = '''
    <start_of_turn>user
    You are a thoughtful analyst tasked with reviewing a piece of writing and identifying sentences that directly support, explain, or relate to a specific idea.
    Your job is to extract exact sentences from the original content that are semantically related to the provided idea. These may reinforce the idea, give examples, expand on it, or express it in different words.
    Do not leave out any context that helps explain the sentences.
    Here is the original content:
    <original_content>
    {content}
    </original_content>
    And here is the target idea:
    <idea>
    {target_idea}
    </idea>

    Return the matching sentences in this JSON format STRICTLY:

    <format>
    ```json
    {{"related": [
    "...",
    "...",
    ...
    ]
    }}
    ```
    </format>

    Only include exact sentences from the original content. If no sentences match, return an empty list.
    <end_of_turn>
    <start_of_turn>model
    '''
        response = ollama.chat(
            model='qwen3:4b',
            messages=[
                {
                    "role": "user",
                    "content": extract_template.format(content=text, target_idea=idea)
                }
            ]
        )
        response = response['message']['content']
        # print(f'Extracted related sentences:\n{response}\n')
        chunk = ""
        try:
            # Look for JSON content between triple backticks
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If not found, try to extract the entire JSON object
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    print(f"No JSON found in response: {response}")
                    return ""
            
            parsed_feedback = json.loads(json_str)
            sentences = parsed_feedback['related']
            chunk = " ".join(sentences)
            return chunk
        except Exception as e:
            print(f"Error parsing feedback JSON: {e}")
            print(f"Response was: {response}")
            # Create a default feedback structure if parsing fails
            return ""

    
    def split_text(self, text):
        """
        Splits text into chunks using RecursiveCharacterTextSplitter.
        
        :param text: The input string to be split.
        :param chunk_size: Maximum size of each chunk.
        :param chunk_overlap: Number of overlapping characters between chunks.
        :return: List of text chunks.
        """
        chunks = []
        try:
            list_of_ideas = self.write_chunking_plan(text)
            print(f'List of ideas: {list_of_ideas}\n')
            # checked_ideas = self.check_idea_redundancy(list_of_ideas)
            # print(f'Checked ideas: {checked_ideas}\n')
            for idea in list_of_ideas:
                print(f'Extracting chunk for idea: {idea}')
                chunk = self.extract_chunk(idea, text)
                if chunk:
                    print(f'Extracted chunk: {chunk}\n')
                    chunks.append(idea + ". " + chunk)
        except Exception as e:
            print("********************************************** SEMANTIC CHUNKING FAILED **********************************************.")
            print(e)
        
        return chunks

if __name__ == "__main__":
    # Example
    chunker = SemanticChunker()
    text = """ Architectural Thinking I
CS 301: IT Solution Architecture
Week 1
Today’s coverage
Key learning objectives
Understand what is architecture and software qualities.
Describe an architecture using sequence diagram.
Introduction to cloud architectures
2
Understanding Requirements is Key
3
Problems that happen before …
4
(Availability)
Where are the components
redundancies in the architecture to
prevent single point of failure?
Straits Times 28 Feb 2018
(Security)
How does internet surfing separation help?
What are the trade-offs?
Problems that happen before …
5
6
How is Architecture Defined?
“The fundamental
organization of a
system embodied in its
components,
their relationships to
each other and to
the environment and
the principles
guiding its design and
evolution.”
- IEEE Std. 1471 definition
6
Discussion
7
What are the components in this photo?  
What are the relationships defined in this photo?
What is the environment depicted in this photo?
Based on your assumptions, what are the key architectural
principles?
Information System Architecture
8
Components
Environment
and Principles
Relationships
Software applications, servers, routers, firewalls ..
How are they integrated
Data centre, cloud environment
How does it relate to the architecture of an information system?
Winchester Mystery House
9
Winchester House is a mansion owned by Sarah Winchester at San
Jose, California. Construction commenced in 1884 and proceeded
daily till her death on Sep 1922.
It was built without architectural planning and odd design details
Describing your architecture
10
How do you describe your ”good” architecture design?
IEEE 42010 Standard
11
Adapted from ANSI/IEEE Standard 42010
12
Architecture Requirements
Functional Requirements (what it must do)
Quality Attributes
(how well)
Constraints
(arbitrary design requirements)
Architecture
Requirements
©Anthony J. Lattanze 2004
“1 click checkout”
“Search for all classes student is allowed to take”
“Must be built on WebSphere technology”
“Have to be at least three-tiered”
“Serve 100 users”
“99.5% availability”
13
ISO 25010 Quality Attributes
Specific concern about system behavior (aka non - functional
requirements)
Typically addressed by one or more views per quality attribute
ISO 25010
13
ISO/IEC 25010 (Online) http://iso25000.com/index.php/en/
iso-25000-standards/iso-25010
14
Architecture Requirements
Functional Requirements (what it must do)
Quality Attributes
(how well)
Constraints
(arbitrary design requirements)
Architecture
Requirements
©Anthony J. Lattanze 2004
“1 click checkout”
“Search for all classes student is allowed to take”
“Must be built on AWS Technology”
“Have to be at least three-tiered”
“Serve 100 users”
“99.5% availability”
Are these stated qualities “good” enough?
Expressing Quality Attribute – Scenario
Expressing QAs
“Fast” is not a requirement, “Handles 100 users” are not specific
Be specific about the action, condition and environment. –
measurable, testable.
“100 concurrent logins with response time of <10 seconds each
during normal operation”
“Homepage is unavailable < 1 min during server failover
operation”
Be aware of the trade-offs
when we design for one quality, do we unintentionally impact
another?
Can you think of one trade-off?
15
Architectural Significant Requirements (ASR)
During design, you will not be able to prototype or validate ALL
features.
An architecturally significant requirement, or ASR, is any
requirement that strongly influences our choice of structures for the
architecture.
Influential Functional Requirements – Features that require special
attention in the architecture. E.g., features used 80% of the time,
features that require end-to-end architecture components.
Quality Attributes - Externally visible properties that characterize
how the system operates in a specific context. Refer to earlier
ISO25010 quality attributes.
Constraints- Unchangeable design decisions, usually given,
sometimes chosen. E.g., must be built on AWS technology.
Other Influencers - Time, knowledge, experience, skills, office
politics, your own geeky biases, and all the other stuff that sways
your decision making.
What’s not? For example, login use case when the user credentials
are verified against a database. (assuming the database is validated
in other requirements)
16
View Diagrams
2 main types we will use in this course
Network diagram for network structure
Sequence diagram for data/control flow (Data-flow, activity, state
diagrams are also popular)
We will use them for
Communication
Analysis
17
Execution flow of the system
What are the modules/components and their interactions?
Top-down sequence
Possible initiators / actors
Human user, External system, Timer
All other arrows must be caused by a previous arrow
A synchronous message always have a return message
An asynchronous message does not have a return message
Sequence Diagram
18
Sequence Diagram – what’s wrong?
Player 1
Portal
Game App
DB
Move Piece
Player 2
Your turn
Save Board
Valid?
Saved
New Board
19
Exercise
20
Requirements + Sequence Diagram
Introduction to Cloud Computing
21
What is Cloud Computing?
Who is responsible for the security?
Cloud Architecture - Service Models
22
Infrastructure-as-a-service
Platform-as-a-service
Software-as-a-service
Cloud Architecture - Deployment Models
23
On-site Private Cloud
Outsourced Private Cloud
On-Site Community Cloud
Outsourced Community Cloud
Cloud Architecture - Deployment Models
24
Public Cloud
Hybrid Cloud
Amazon Web Services (AWS) Service Stack
25
Infrastructure
Regions
Edge locations
Availability Zones
Foundation
Services
Compute
(Virtual, auto scaling and load balancing)
Networking
Applications
Virtual Desktops
Collaboration and Sharing
Platform Services
Caching
Relational
No SQL
Real time
Cluster computing
Data workflows
Data warehouse
Queuing
Orchestration
App streaming
Transcoding
Email
Search
Containers
Dev/ops tools
Resource templates
Usage tracking
Monitoring and logs
Identity
Sync
Mobile Analytics
Notifications
Databases
Analytics
App Services
Deployment and Management
Mobile Services
Storage
(Object, block and archive)
AWS Services (Compute, Storage and Database) – A Sample List
Amazon EC2
Amazon Elastic Container Registry (Amazon ECR)
Amazon Elastic Container Service (Amazon ECS)
AWS Elastic Beanstalk
AWS Lambda
Compute Services
Storage Services
Database Services
Amazon S3
Amazon S3 Glacier
Amazon EBS
Amazon
EFS
Amazon Relational Database Service (Amazon RDS)
Amazon DynamoDB
Amazon Aurora
Amazon Redshift
AWS Shared Responsibility Model
27
Responsibility: Security of / in the Cloud
28
Question - Who is responsible for maintaining security?
29
Consider these situations. Who is responsible? AWS or Customer
Upgrades and patches to the OS on the EC2 instance?
Physical security of the data center?
Virtualization Infrastructure?
EC2 security group settings?
Configuration of applications that run on the EC2 instance?
Oracle upgrades or patches if Oracle instance runs as an Amazon
RDS instance?
Oracle upgrades or patches if Oracle runs on an EC2 instance?
S3 bucket access configuration?
Amazon Simple Storage Service (Amazon S3)
Amazon EC2
Virtual Private Cloud (VPC)
Oracle instance
AWS Cloud
AWS Global Infrastructure
Summary
Definition of Architecture
Components, Relationships, Environment, Principles
Sequence Diagram
Answer the stakeholders’ questions
No one picture is best for all questions
There are many other types of diagrams!!
Requirements include:
Features to be delivered, constraints on technologies to be used,
quality attributes
Good designs meet not just functional requirements but also
achieves the qualities and works within the constraints. Focus on
the architectural significant requirements.
Cloud Computing
Service and Deployment Models
AWS Service Stack and Responsibility Model
30




http://www.iso-architecture.org/ieee-1471/defining-architecture.html
What are the components in this photo?  
 
What are the relationships defined in this photo? 
  
What is the environment depicted in this photo?
 
Based on your assumptions, what are the key architectural principles? 

Components or “Architectural Elements”
Software Modules/Packages, Hardware Machines, Network Devices…
Detailed objects within modules/packages are generally not required
Some things to look for:
Global, local, mobile access- Often lead to bandwidth, latency, security issues
Distributed (How many machines?)
Future changes, variations and compatibility

Relationships or “Interfaces” or “Connectors”
How are the pieces put together? *overlap with enterprise integration*
How do they communicate or talk to each other?
Some things to look for:
Data flow between high level components
Integration with “external” systems

Environment 
What are the existing pieces that needs to be integrated?
Can be complex, uncertain or poorly structured that will impact your architectural decisions

Principles 
What are the decisions made for your structure?
Have to focus on the architectural significance decisions. 
Not easy but that is why architectural thinking is different from software programming and low level software design.


https://winchestermysteryhouse.com/

https://www.youtube.com/watch?v=5DVEPBlZknk




ISO/IEC/IEEE 42010: Conceptual Model (iso-architecture.org)

The original version is IEEE 1471.

A system’s environment, or context, can influence that system. The environment can include other systems that interact with the system of interest, either directly via interfaces or indirectly in other ways. The environment determines the boundaries that define the scope of the system of interest relative to other systems.

A system has one or more stakeholders. Each stakeholder typically has interests in, or concerns relative to, that system.

Concerns are those interests which pertain to the system’s development, its operation or any other aspects that are critical or otherwise important to one or more stakeholders. Concerns include system considerations such as performance, reliability, security, distribution, and evolvability.

A system exists to fulfill one or more missions in its environment. A mission is a use or operation for which a system is intended by one or more stakeholders to meet some set of objectives.
Every system has an architecture, whether understood or not; whether recorded or conceptual. An architecture can be recorded by an architectural description.

An architectural description is organized into one or more constituents called (architectural) views. Each view addresses one or more of the concerns of the system stakeholders. A view is a partial expression of a system’s architecture with respect to a particular viewpoint.

A viewpoint establishes the conventions by which a view is created, depicted and analyzed. In this way, a view conforms to a viewpoint. The viewpoint determines the languages (including notations, model, or product types) to be used to describe the view, and any associated modeling methods or analysis techniques to be applied to these representations of the view. These languages and techniques are used to yield results relevant to the concerns addressed by the viewpoint.

An architectural description selects one or more viewpoints for use. The selection of viewpoints is typically based on consideration of the stakeholders to whom the AD is addressed and their concerns. A viewpoint definition may originate with an AD, or it may have been defined elsewhere (a library viewpoint).

A view may consist of one or more architectural models. Each such architectural model is developed using the methods established by its associated architectural viewpoint. An architectural model may participate in more than one view.

Architecture Rationale records the explanation, justification or reasoning about Architecture Decisions that have been made and architectural alternatives not chosen.

Constraints are design and implementation decisions already made.  For example:
Organizational skills (java/C#, oracle/MySQL, SAP/PeopleSoft, web/client-server)
Standards (MQ, HTTP/HTTPs, mp3, pdf)
Policies and Regulations
Partnerships (clients, suppliers)

Constraints from one architect may flow to another.
Data, network, interfaces, etc.

Functional Suitability
This characteristic represents the degree to which a product or system provides functions that meet stated and implied needs when used under specified conditions. This characteristic is composed of the following sub-characteristics:
Functional completeness - Degree to which the set of functions covers all the specified tasks and user objectives.
Functional correctness - Degree to which a product or system provides the correct results with the needed degree of precision.
Functional appropriateness - Degree to which the functions facilitate the accomplishment of specified tasks and objectives.

Performance efficiency
This characteristic represents the performance relative to the amount of resources used under stated conditions. This characteristic is composed of the following sub-characteristics:
Time behaviour - Degree to which the response and processing times and throughput rates of a product or system, when performing its functions, meet requirements.
Resource utilization - Degree to which the amounts and types of resources used by a product or system, when performing its functions, meet requirements.
Capacity - Degree to which the maximum limits of a product or system parameter meet requirements.

Compatibility
Degree to which a product, system or component can exchange information with other products, systems or components, and/or perform its required functions while sharing the same hardware or software environment. This characteristic is composed of the following sub-characteristics:
Co-existence - Degree to which a product can perform its required functions efficiently while sharing a common environment and resources with other products, without detrimental impact on any other product.
Interoperability - Degree to which two or more systems, products or components can exchange information and use the information that has been exchanged.

Usability
Degree to which a product or system can be used by specified users to achieve specified goals with effectiveness, efficiency and satisfaction in a specified context of use. This characteristic is composed of the following sub-characteristics:
Appropriateness recognizability - Degree to which users can recognize whether a product or system is appropriate for their needs.
Learnability - Degree to which a product or system can be used by specified users to achieve specified goals of learning to use the product or system with effectiveness, efficiency, freedom from risk and satisfaction in a specified context of use.
Operability - Degree to which a product or system has attributes that make it easy to operate and control.
User error protection. Degree to which a system protects users against making errors.
User interface aesthetics - Degree to which a user interface enables pleasing and satisfying interaction for the user.
Accessibility - Degree to which a product or system can be used by people with the widest range of characteristics and capabilities to achieve a specified goal in a specified context of use.

Reliability
Degree to which a system, product or component performs specified functions under specified conditions for a specified period of time. This characteristic is composed of the following sub-characteristics:
Maturity - Degree to which a system, product or component meets needs for reliability under normal operation.
Availability - Degree to which a system, product or component is operational and accessible when required for use.
Fault tolerance - Degree to which a system, product or component operates as intended despite the presence of hardware or software faults.
Recoverability - Degree to which, in the event of an interruption or a failure, a product or system can recover the data directly affected and re-establish the desired state of the system.

Security
Degree to which a product or system protects information and data so that persons or other products or systems have the degree of data access appropriate to their types and levels of authorization. This characteristic is composed of the following sub-characteristics:
Confidentiality - Degree to which a product or system ensures that data are accessible only to those authorized to have access.
Integrity - Degree to which a system, product or component prevents unauthorized access to, or modification of, computer programs or data.
Non-repudiation - Degree to which actions or events can be proven to have taken place so that the events or actions cannot be repudiated later.
Accountability - Degree to which the actions of an entity can be traced uniquely to the entity.
Authenticity - Degree to which the identity of a subject or resource can be proved to be the one claimed.

Maintainability
This characteristic represents the degree of effectiveness and efficiency with which a product or system can be modified to improve it, correct it or adapt it to changes in environment, and in requirements. This characteristic is composed of the following sub-characteristics:
Modularity - Degree to which a system or computer program is composed of discrete components such that a change to one component has minimal impact on other components.
Reusability - Degree to which an asset can be used in more than one system, or in building other assets.
Analysability - Degree of effectiveness and efficiency with which it is possible to assess the impact on a product or system of an intended change to one or more of its parts, or to diagnose a product for deficiencies or causes of failures, or to identify parts to be modified.
Modifiability - Degree to which a product or system can be effectively and efficiently modified without introducing defects or degrading existing product quality.
Testability - Degree of effectiveness and efficiency with which test criteria can be established for a system, product or component and tests can be performed to determine whether those criteria have been met.

Portability
Degree of effectiveness and efficiency with which a system, product or component can be transferred from one hardware, software or other operational or usage environment to another. This characteristic is composed of the following sub-characteristics:
Adaptability - Degree to which a product or system can effectively and efficiently be adapted for different or evolving hardware, software or other operational or usage environments.
Installability - Degree of effectiveness and efficiency with which a product or system can be successfully installed and/or uninstalled in a specified environment.
Replaceability - Degree to which a product can replace another specified software product for the same purpose in the same environment.



Trade-offs – When we design for one quality, we may unintentionally negatively impact another quality. For example, two factor authentication for security can trade-off the usability of the system.
The Rational Edge -- November 2001 -- Capturing Architectural Requirements (researchgate.net)

Architecturally Significant Requirements (iasaglobal.org)

Preethu-Roese-Maya-Daneva-Jane-Cleland-Huang_et_al_RE2015_Architecture_Significant_Functional_Requirements.pdf

Architecturally significant requirements - Wikipedia
Deployment diagram for system structure is covered in earlier terms but removed this term.
Activation bar is optional.

Note that the components are at the architectural level. E.g., database, web server, app server etc. We are not interested in the classes yet which can be too messy to draw out.



http://www.nist.gov/itl/csd/cloud-102511.cfm

https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-145.pdf

Cloud computing is a model for enabling ubiquitous, convenient, on-demand network access to a shared pool of configurable computing resources (e.g., networks, servers, storage, applications, and services) that can be rapidly provisioned and released with minimal management effort or service provider interaction. This cloud model is composed of five essential characteristics, three service models, and four deployment models. 

Essential Characteristics: 

On-demand self-service. A consumer can unilaterally provision computing capabilities, such as server time and network storage, as needed automatically without requiring human interaction with each service provider. 

Broad network access. Capabilities are available over the network and accessed through standard mechanisms that promote use by heterogeneous thin or thick client platforms (e.g., mobile phones, tablets, laptops, and workstations).

Resource pooling. The provider’s computing resources are pooled to serve multiple consumers using a multi-tenant model, with different physical and virtual resources dynamically assigned and reassigned according to consumer demand. There is a sense of location independence in that the customer generally has no control or knowledge over the exact location of the provided resources but may be able to specify location at a higher level of abstraction (e.g., country, state, or datacenter). Examples of resources include storage, processing, memory, and network bandwidth. 

Rapid elasticity. Capabilities can be elastically provisioned and released, in some cases automatically, to scale rapidly outward and inward commensurate with demand. To the consumer, the capabilities available for provisioning often appear to be unlimited and can be appropriated in any quantity at any time.

Measured service. Cloud systems automatically control and optimize resource use by leveraging a metering capability1 at some level of abstraction appropriate to the type of service (e.g., storage, processing, bandwidth, and active user accounts). Resource usage can be monitored, controlled, and reported, providing transparency for both the provider and consumer of the utilized service. 

Service Models: Software as a Service (SaaS). The capability provided to the consumer is to use the provider’s applications running on a cloud infrastructure. The applications are accessible from various client devices through either a thin client interface, such as a web browser (e.g., web-based email), or a program interface. The consumer does not manage or control the underlying cloud infrastructure including network, servers, operating systems, storage, or even individual application capabilities, with the possible exception of limited userspecific application configuration settings. 

Platform as a Service (PaaS). The capability provided to the consumer is to deploy onto the cloud infrastructure consumer-created or acquired applications created using programming languages, libraries, services, and tools supported by the provider.3 The consumer does not manage or control the underlying cloud infrastructure including network, servers, operating systems, or storage, but has control over the deployed applications and possibly configuration settings for the application-hosting environment.

Infrastructure as a Service (IaaS). The capability provided to the consumer is to provision processing, storage, networks, and other fundamental computing resources where the consumer is able to deploy and run arbitrary software, which can include operating systems and applications. The consumer does not manage or control the underlying cloud infrastructure but has control over operating systems, storage, and deployed applications; and possibly limited control of select networking components (e.g., host firewalls).

Deployment Models:
Private cloud. The cloud infrastructure is provisioned for exclusive use by a single organization comprising multiple consumers (e.g., business units). It may be owned, managed, and operated by the organization, a third party, or some combination of them, and it may exist on or off premises. 

Community cloud. The cloud infrastructure is provisioned for exclusive use by a specific community of consumers from organizations that have shared concerns (e.g., mission, security requirements, policy, and compliance considerations). It may be owned, managed, and operated by one or more of the organizations in the community, a third party, or some combination of them, and it may exist on or off premises. 

Public cloud. The cloud infrastructure is provisioned for open use by the general public. It may be owned, managed, and operated by a business, academic, or government organization, or some combination of them. It exists on the premises of the cloud provider. 

Hybrid cloud. The cloud infrastructure is a composition of two or more distinct cloud infrastructures (private, community, or public) that remain unique entities, but are bound together by standardized or proprietary technology that enables data and application portability (e.g., cloud bursting for load balancing between clouds).

http://www.nist.gov/itl/csd/cloud-102511.cfm

http://www.nist.gov/itl/csd/cloud-102511.cfm

Compute Services
Amazon Elastic Compute Cloud (Amazon EC2):
Virtual computing environment in the cloud
In terms of deployment diagram, AWS manage the device node and gives you the OS execution environment.
You can update the OS execution environment, install new software onto the OS and execute your programs.

AWS Lambda:
Fully managed serverless compute
In terms of deployment diagram, AWS manage the device node, OS execution environment and application/framework execution environment (e.g., JDK). 
You execute your programs within their application/framework execution environment.

Amazon Elastic Container Services (ECS):
Amazon Elastic Container Service (Amazon ECS) is a highly scalable, high-performance container management service that supports Docker containers. 
Amazon ECS enables you to easily run applications on a managed cluster of Amazon EC2 instances.

Amazon Elastic Container Registry (Amazon ECR) 
is a fully managed Docker container registry that makes it easy for developers to store, manage, and deploy Docker container images. 
It is integrated with Amazon ECS, so you can store, run, and manage container images for applications that run on Amazon ECS.

Amazon Elastic Container Service for Kubernetes (EKS):
Run Kubernetes without managing Kubernetes clusters

AWS Elastic Beanstalk:
Quickly deploys, scales, and manages web apps
No charge for Elastic Beanstalk – pay only for the underlying AWS services used

Storage Services
Amazon EBS is persistent, mountable storage that can be mounted as a device to an Amazon EC2 instance. Amazon EBS can be mounted to an Amazon EC2 instance only
within the same Availability Zone. Only one Amazon EC2 instance at a time can mount an Amazon EBS volume.

Amazon EFS is a shared file system that multiple Amazon EC2 instances can mount at the same time.

Amazon S3 is persistent storage where each file becomes an object and is available through a Uniform Resource Locator (URL); it can be accessed from anywhere.

Amazon S3 Glacier is for cold storage for data that is not accessed frequently (for example, when you need long-term data storage for archival or compliance reasons).

Database Services
Amazon RDS is a managed service that sets up and operates a relational database in the cloud.

DynamoDB is a fast and flexible NoSQL database service for all applications that need consistent, single-digit-millisecond latency at any scale.

Amazon Redshift is a fast, fully managed data warehouse that makes it simple and cost-effective to analyze all your data by using standard SQL and your existing business intelligence (BI) tools.
It enables you to run complex analytic queries against petabytes of structured data by using sophisticated query optimization, columnar storage on high-performance local disks, and massively parallel query execution. Most results come back in seconds.

Amazon Aurora is a MySQL- and PostgreSQL-compatible relational database that is built for the cloud. It combines the performance and availability of high-end commercial databases with the simplicity and cost-effectiveness of open-source databases.


AWS operates, manages, and controls the components from the software virtualization layer down to the physical security of the facilities where AWS services operate.

AWS is responsible for protecting the infrastructure that runs all the services that are offered in the  AWS Cloud. This infrastructure is composed of the hardware, software, networking, and facilities that run the AWS Cloud services.

The customer is responsible for the encryption of data at rest and data in transit. The customer should also ensure that the network is configured for security and that security credentials and logins are managed safely. Additionally, the customer is responsible for the configuration of security groups and the configuration of the operating system that run on compute instances that they launch (including updates and security patches). 


https://aws.amazon.com/compliance/shared-responsibility-model/

AWS is responsible for security of the cloud

Under the AWS shared responsibility model, AWS operates, manages, and controls the components from the bare metal host operating system and hypervisor virtualization layer 
down to the physical security of the facilities where the services operate. It means that AWS is responsible for protecting the global infrastructure that runs all the services that are  offered in the AWS Cloud. The global infrastructure includes AWS Regions, Availability Zones, and edge locations.


AWS and the customer share security responsibilities–
AWS is responsible for security of the cloud
Customer is responsible for security in the cloud
AWS is responsible for protecting the infrastructure —including hardware, software, networking, and facilities— that run AWS Cloud services
For services that are categorized as infrastructure as a service (IaaS), the customer is responsible for performing necessary security configuration and management tasks
For example, guest OS updates and security patches, firewall, security group configurations

AWS is responsible for the physical infrastructure that hosts your resources, including:
Physical security of data centers with controlled, need-based access; located in nondescript facilities, with 24/7 security guards; two-factor authentication; access logging and review; video surveillance; and disk degaussing and destruction.
Hardware infrastructure, such as servers, storage devices, and other appliances that AWS relies on.
Software infrastructure, which hosts operating systems, service applications, and virtualization software.
Network infrastructure, such as routers, switches, load balancers, firewalls, and cabling.  AWS also continuously monitors the network at external boundaries, secures access points, and provides redundant infrastructure with intrusion detection.

While the cloud infrastructure is secured and maintained by AWS, customers are responsible for security of everything they put in the cloud. 
The customer is responsible for what is implemented by using AWS services and for the applications that are connected to AWS. The security steps that you must take depend on the services that you use and the complexity of your system.

Customer responsibilities include selecting and securing any instance operating systems,  securing the applications that are launched on AWS resources, security group configurations,  firewall configurations, network configurations, and secure account management.  When customers use AWS services, they maintain complete control over their content. Customers  are responsible for managing critical content security requirements, including: 
What content they choose to store on AWS
Which AWS services are used with the content
In what country that content is stored
The format and structure of that content and whether it is masked, anonymized, or encrypted
Who has access to that content and how those access rights are granted, managed, and revoked




Consider the case where a customer uses the AWS services and resources that are shown here. Who is responsible for maintaining security? AWS or the customer?
 
The customer uses Amazon Simple Storage Service (Amazon S3) to store data. The customer configured a virtual private cloud (VPC) with Amazon Virtual Private Cloud (Amazon VPC). The EC2 instance and the Oracle database instance that they created both run in the VPC.

In this example, the customer must manage the guest operating system (OS) that runs on the EC2 instance. Over time, the guest OS will need to be upgraded and have security patches applied. Additionally, any application software or utilities that the customer installed on the Amazon EC2 instance must also be maintained. The customer is responsible for configuring the AWS firewall (or security group) that is applied to the Amazon EC2 instance. The customer is also responsible for the VPC configurations that specify the network conditions in which the Amazon EC2 instance runs. These tasks are the same security tasks that IT staff would perform, no matter where their servers are located. 

The Oracle instance in this example provides an interesting case study in terms of AWS or customer responsibility. If the database runs on an EC2 instance, then it is the customer's responsibility to apply Oracle software upgrades and patches. However, if the database runs as an Amazon RDS instance, then it is the responsibility of AWS to apply Oracle software upgrades and patches. Because Amazon RDS is a managed database offering, time-consuming database administration tasks—which include provisioning, backups, software patching, monitoring, and hardware scaling—are handled by AWS. To learn more, see Best Practices for Running Oracle Database on AWS for details.


Upgrades and patches to the operating system on the EC2 instance?
ANSWER: The customer
Physical security of the data center?
ANSWER: AWS
Virtualization infrastructure?
ANSWER: AWS
EC2 security group settings?
ANSWER: The customer
Configuration of applications that run on the EC2 instance?
ANSWER: The customer
Oracle upgrades or patches If the Oracle instance runs as an Amazon RDS instance?
ANSWER: AWS
Oracle upgrades or patches If Oracle runs on an EC2 instance?
ANSWER: The customer
S3 bucket access configuration?
ANSWER: The customer"""
    
    
    chunks = chunker.split_text(text)
    print("Semantic Chunks:")
    print("\n".join(chunks))
    # for i, chunk in enumerate(chunks, 1):
    #     print(f"{i}. {chunk}")
# Example usage of the SemanticChunker class