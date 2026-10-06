import json
from pathlib import Path


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKING_DIR = (
    PROJECT_ROOT
    / "data"
    / "working"
    / "member3"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member3_knowledge.json"
)


# =========================================================
# Locate Member 3 source files
# =========================================================

def find_source(fragment: str) -> str:
    """
    Find a Member 3 source file using part of its filename.

    This avoids problems caused by slightly shortened filenames
    or .txt/.md differences.
    """

    fragment = fragment.lower()

    for path in WORKING_DIR.iterdir():

        if (
            path.is_file()
            and fragment in path.name.lower()
        ):
            return path.name

    raise FileNotFoundError(
        f"Could not find Member 3 source file containing: "
        f"{fragment}"
    )


# =========================================================
# Validate source files
# =========================================================

def validate_source_files():

    expected_fragments = [
        "academic integrity",
        "admission and programme",
        "assessment, progression",
        "faculty and department",
        "fees, student support",
        "final year project",
        "general contact",
        "important university",
        "registration and student conduct",
        "student support services",
        "university examinations",
        "university regulations",
    ]

    print("Validating Member 3 source files...")

    for fragment in expected_fragments:

        filename = find_source(fragment)

        source_path = (
            WORKING_DIR
            / filename
        )

        if source_path.stat().st_size == 0:

            raise ValueError(
                f"Source file is empty: {filename}"
            )

        print(
            f"OK: {filename}"
        )


# =========================================================
# Standard record creator
# =========================================================

def make_record(
    record_id,
    category,
    subcategory,
    title,
    description,
    source_file,
    source_url="",
    faculty="University of Mauritius",
    academic_year=""
):

    return {
        "ID": record_id,
        "Category": category,
        "Subcategory": subcategory,
        "Title": title,
        "Faculty": faculty,
        "Academic Year": academic_year,
        "Description": description,
        "Source URL": source_url,
        "Source File": source_file,
        "Verified Status": "Member 3 source-backed"
    }


# =========================================================
# Build structured Member 3 knowledge
# =========================================================

def build_member3_records():

    sources = {

        "integrity":
            find_source(
                "academic integrity"
            ),

        "admission":
            find_source(
                "admission and programme"
            ),

        "assessment":
            find_source(
                "assessment, progression"
            ),

        "faculty":
            find_source(
                "faculty and department"
            ),

        "fees":
            find_source(
                "fees, student support"
            ),

        "fyp":
            find_source(
                "final year project"
            ),

        "general_contact":
            find_source(
                "general contact"
            ),

        "services":
            find_source(
                "important university"
            ),

        "conduct":
            find_source(
                "registration and student conduct"
            ),

        "support":
            find_source(
                "student support services"
            ),

        "examinations":
            find_source(
                "university examinations"
            ),

        "regulations":
            find_source(
                "university regulations"
            ),
    }


    records = [

        # =================================================
        # CONTACT INFORMATION
        # =================================================

        make_record(
            "M3_CONTACT_001",
            "Contact Information",
            "General Contact",
            "University of Mauritius General Contact",
            (
                "The University of Mauritius is located at "
                "Réduit, 80837, Mauritius. "
                "Telephone: (230) 403 7400. "
                "Fax: (230) 454 9642. "
                "General email contacts are "
                "registrar@uom.ac.mu and admission@uom.ac.mu."
            ),
            sources["general_contact"],
            "https://www.uom.ac.mu/"
        ),

        make_record(
            "M3_CONTACT_002",
            "Contact Information",
            "Admissions",
            "Admissions and Student Records Office",
            (
                "The Admissions and Student Records Office "
                "(ASRO) supports current and prospective students "
                "with enrolment, programme enquiries, admissions, "
                "fees, student progress, completion, discipline "
                "and complaints. It is located at the Professor "
                "Sir Edouard Lim Fat Engineering Tower, "
                "University of Mauritius, Réduit. "
                "Telephone numbers include (230) 403 7872, "
                "(230) 403 7797 and (230) 403 7426. "
                "Admissions email: admission@uom.ac.mu."
            ),
            sources["services"],
            (
                "https://www.uom.ac.mu/index.php/"
                "study-at-uom/admissions"
            )
        ),

        make_record(
            "M3_CONTACT_003",
            "Contact Information",
            "Examinations Office",
            "University of Mauritius Examinations Office",
            (
                "The Examinations Office is located on the "
                "1st Floor of the CPDL Building at the "
                "University of Mauritius, Réduit. "
                "Telephone numbers include (230) 403 7645, "
                "(230) 403 7635, (230) 403 7632 and "
                "(230) 403 7400 Ext. 2145. "
                "Fax: (230) 465 6977."
            ),
            sources["services"]
        ),

        make_record(
            "M3_CONTACT_004",
            "Contact Information",
            "Library Contact",
            "University of Mauritius Library Contact",
            (
                "The University of Mauritius Library is located "
                "at the University of Mauritius, Réduit. "
                "Telephone: (230) 403 7915. "
                "Fax: (230) 464 0905."
            ),
            sources["services"],
            (
                "https://www.uom.ac.mu/index.php/"
                "library/contact-information"
            )
        ),

        make_record(
            "M3_CONTACT_005",
            "Contact Information",
            "International Affairs",
            "International Affairs Office Contact",
            (
                "International students may contact the "
                "International Affairs Office. "
                "General international enquiries may use "
                "isao@uom.ac.mu. "
                "Applicants with questions about an existing "
                "application may use studyatuom@uom.ac.mu. "
                "The supplied contact information also lists "
                "telephone numbers including +230 403 7440 "
                "and +230 5902 5055."
            ),
            sources["services"],
            (
                "https://www.uom.ac.mu/index.php/"
                "iao-contact-us"
            )
        ),


        # =================================================
        # FACULTIES AND DEPARTMENTS
        # =================================================

        make_record(
            "M3_FAC_001",
            "Faculty Information",
            "Faculty",
            "Faculty of Agriculture",
            (
                "The Faculty of Agriculture has two departments: "
                "Department of Agricultural Production & Systems "
                "(APS), and Department of Agricultural & Food "
                "Science (AFS)."
            ),
            sources["faculty"],
            "https://uom.ac.mu/FOA/",
            faculty="Faculty of Agriculture"
        ),

        make_record(
            "M3_FAC_002",
            "Faculty Information",
            "Faculty",
            "Faculty of Engineering",
            (
                "The Faculty of Engineering has five departments: "
                "Chemical and Environmental Engineering, "
                "Civil Engineering, Electrical and Electronic "
                "Engineering, Applied Sustainability and "
                "Enterprise Development, and Mechanical and "
                "Production Engineering."
            ),
            sources["faculty"],
            "https://www.uom.ac.mu/foe/",
            faculty="Faculty of Engineering"
        ),

        make_record(
            "M3_FAC_003",
            "Faculty Information",
            "Faculty",
            (
                "Faculty of Information, Communication "
                "and Digital Technologies"
            ),
            (
                "The Faculty of Information, Communication and "
                "Digital Technologies has three departments: "
                "Digital Technologies, Information and "
                "Communication Technologies, and Software and "
                "Information Systems. The Department of "
                "Information and Communication Technologies "
                "offers programmes including BSc (Hons) "
                "Computer Science and BSc (Hons) Cyber Security."
            ),
            sources["faculty"],
            "https://www.uom.ac.mu/foicdt/",
            faculty="FOICDT"
        ),

        make_record(
            "M3_FAC_004",
            "Faculty Information",
            "Faculty",
            "Faculty of Law & Management",
            (
                "The Faculty of Law & Management has three "
                "departments: Finance & Accounting, Law, "
                "and Management."
            ),
            sources["faculty"],
            "https://www.uom.ac.mu/flm/",
            faculty="Faculty of Law & Management"
        ),

        make_record(
            "M3_FAC_005",
            "Faculty Information",
            "Faculty",
            "Faculty of Medicine and Health Sciences",
            (
                "The Faculty of Medicine and Health Sciences "
                "has two departments: Department of Medicine "
                "and Department of Health Sciences."
            ),
            sources["faculty"],
            "https://www.uom.ac.mu/fmhs/",
            faculty="Faculty of Medicine and Health Sciences"
        ),

        make_record(
            "M3_FAC_006",
            "Faculty Information",
            "Faculty",
            "Faculty of Science",
            (
                "The Faculty of Science has four departments: "
                "Biological Sciences and Ocean Studies, "
                "Chemistry, Mathematics and Physics."
            ),
            sources["faculty"],
            "https://www.uom.ac.mu/fos/",
            faculty="Faculty of Science"
        ),

        make_record(
            "M3_FAC_007",
            "Faculty Information",
            "Faculty",
            "Faculty of Social Sciences & Humanities",
            (
                "The Faculty of Social Sciences & Humanities "
                "has five departments: Economics and Statistics, "
                "English Studies, French Studies, History and "
                "Political Science, and Social Studies."
            ),
            sources["faculty"],
            "https://www.uom.ac.mu/fssh/",
            faculty="Faculty of Social Sciences & Humanities"
        ),


        # =================================================
        # STUDENT SUPPORT
        # =================================================

        make_record(
            "M3_SUPPORT_001",
            "Student Support",
            "Students Union",
            "University of Mauritius Students' Union",
            (
                "The Students' Union is the official organisation "
                "of students at the University of Mauritius. "
                "It represents students and promotes student "
                "welfare and social, cultural and educational "
                "activities. The office is located on the "
                "1st Floor of the Student Centre. "
                "Telephone: 403 7400 Ext. 4016. "
                "Email: presidentsu@uom.ac.mu."
            ),
            sources["support"]
        ),

        make_record(
            "M3_SUPPORT_002",
            "Student Support",
            "Health and Recreation",
            "First Aid and Sports Services",
            (
                "The University provides First Aid services "
                "for students and staff, including general and "
                "emergency medical assistance under medical "
                "supervision. The University also provides "
                "sports and recreational facilities and "
                "activities for students."
            ),
            sources["support"],
            "https://www.uom.ac.mu/students/"
        ),

        make_record(
            "M3_SUPPORT_003",
            "Student Support",
            "Counselling",
            "Student Counselling",
            (
                "The University provides counselling support "
                "for students experiencing difficulties that "
                "may affect their studies, including study "
                "difficulties, personal problems, trauma and "
                "peer conflicts. Students may be referred "
                "through ASRO to appropriate professionals. "
                "Confidentiality is taken into consideration."
            ),
            sources["support"]
        ),

        make_record(
            "M3_SUPPORT_004",
            "Student Support",
            "Financial Assistance",
            "Student Welfare and Financial Assistance",
            (
                "The University has a Student Welfare Office "
                "and financial assistance schemes for eligible "
                "students. The UoM Students Financial Assistance "
                "Fund is available to eligible full-time, "
                "non-tuition-fee-paying undergraduate students. "
                "The Student Welfare Office is located at "
                "Room 7.14, 7th Floor, Academic Complex, "
                "Tower Block, University of Mauritius, Réduit."
            ),
            sources["support"],
            (
                "https://online.uom.ac.mu/"
                "studentassistance/startpage.aspx"
            )
        ),

        make_record(
            "M3_SUPPORT_005",
            "Student Support",
            "International Students",
            "International Student Support",
            (
                "The International Affairs Office provides "
                "support and information for international "
                "students in areas including admissions, "
                "scholarships, visa, accommodation, medical "
                "insurance, academic calendar, facilities, "
                "counselling, Student Union and payment "
                "facilities. Email: iao@uom.ac.mu. "
                "Telephone: (+230) 403 7587."
            ),
            sources["support"],
            "https://www.uom.ac.mu/index.php/welcome"
        ),


        # =================================================
        # ADMISSION, PROGRAMME STRUCTURE AND FEES
        # =================================================

        make_record(
            "M3_ACAD_001",
            "University Regulations",
            "Admission",
            "Admission Requirements",
            (
                "Applicants must follow the prescribed "
                "University application procedure, submit the "
                "required information and documents within "
                "the specified deadline, satisfy general entry "
                "requirements and satisfy the specific "
                "requirements of the chosen programme."
            ),
            sources["admission"],
            (
                "https://www.uom.ac.mu/index.php/"
                "study-at-uom/current-students/regulations/"
                "undergraduate-postgraduate"
            ),
            academic_year="2026/2027"
        ),

        make_record(
            "M3_ACAD_002",
            "University Regulations",
            "Programme Structure",
            "Programme Structure and Credit System",
            (
                "University regulations define the academic "
                "year, programme of study, modules, credit "
                "system, credit equivalence, contact hours, "
                "programme duration, module types, mode of "
                "delivery and different forms of placement "
                "and practical training."
            ),
            sources["admission"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap2.pdf"
            ),
            academic_year="2026/2027"
        ),

        make_record(
            "M3_ACAD_003",
            "University Regulations",
            "Module Types",
            "Types of Modules",
            (
                "Modules may include core modules, "
                "elective or optional modules, self-study "
                "modules, independent study and audit modules."
            ),
            sources["admission"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap2.pdf"
            ),
            academic_year="2026/2027"
        ),

        make_record(
            "M3_ACAD_004",
            "University Regulations",
            "Credits and Duration",
            "Programme Credits, Duration and NCVTS",
            (
                "The 2026/2027 regulations specify credit "
                "ranges and normal durations according to "
                "qualification type. Examples include "
                "96-106 UoM credits for a 3-year undergraduate "
                "degree and 36-42 UoM credits for a Master's "
                "award. The National Credit Value and Transfer "
                "System (NCVTS) is being implemented from "
                "Academic Year 2026/2027."
            ),
            sources["admission"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap2.pdf"
            ),
            academic_year="2026/2027"
        ),

        make_record(
            "M3_ACAD_005",
            "University Regulations",
            "Fees",
            "University Fees and Charges",
            (
                "University regulations cover general fees, "
                "tuition fees, Students' Welfare Fund, "
                "laboratory fees, clinical fees, placement "
                "fees, dissertation fees, refunds, late module "
                "registration fees, module exemption fees, "
                "re-examination fees, re-registration fees, "
                "review and appeal fees, transcripts, "
                "certificates, student ID cards and graduation "
                "fees. The University may review fees for an "
                "academic year."
            ),
            sources["fees"],
            "https://www.uom.ac.mu/index.php/regulations"
        ),


        # =================================================
        # ASSESSMENT AND PROGRESSION
        # =================================================

        make_record(
            "M3_ASSESS_001",
            "University Regulations",
            "Assessment",
            "Assessment and Continuous Assessment",
            (
                "Assessment may include written examinations, "
                "continuous assessment, laboratory work, "
                "seminars, assignments and class tests. "
                "Continuous assessment is normally weighted "
                "20% to 40% for undergraduate programmes and "
                "30% to 40% for postgraduate programmes, "
                "unless otherwise specified."
            ),
            sources["assessment"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap4.pdf"
            )
        ),

        make_record(
            "M3_ASSESS_002",
            "University Regulations",
            "Academic Progress",
            "GPA, CPA and Prerequisites",
            (
                "The University uses Grade Point Average "
                "(GPA), Cumulative Point Average (CPA) and "
                "Level or Year Point Average calculations. "
                "Modules may also have prerequisites, "
                "pre-requirements or minimum requirements "
                "that must be satisfied before registration."
            ),
            sources["assessment"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap4.pdf"
            )
        ),

        make_record(
            "M3_ASSESS_003",
            "University Regulations",
            "Retakes and Resits",
            "Retakes, Resits and Reassessment",
            (
                "University regulations provide for retake "
                "modules, special retake examinations, resit "
                "examinations, re-submission of dissertations "
                "and assessment of phased-out modules."
            ),
            sources["assessment"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap4.pdf"
            )
        ),

        make_record(
            "M3_ASSESS_004",
            "University Regulations",
            "Progression",
            "Progression, Repeating and Awards",
            (
                "Progression is governed by applicable yearly "
                "or semester regulations and programme-specific "
                "requirements. Regulations also cover repeating "
                "a year, termination of registration and "
                "classification of final awards."
            ),
            sources["assessment"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap4.pdf"
            )
        ),


        # =================================================
        # EXAMINATION REGULATIONS
        # =================================================

        make_record(
            "M3_EXAM_001",
            "University Regulations",
            "Examination Conduct",
            "Examination Conduct and Unauthorised Devices",
            (
                "Students must comply with examination "
                "regulations and instructions from examination "
                "staff and invigilators. Unauthorised devices "
                "are prohibited. A student using an unauthorised "
                "device may be reported for breach of "
                "examination regulations and the device may "
                "be confiscated."
            ),
            sources["examinations"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/Chap6.pdf"
            )
        ),

        make_record(
            "M3_EXAM_002",
            "University Regulations",
            "Medical Absence",
            "Absence from an Examination Due to Illness",
            (
                "If examination absence is due to ill health, "
                "the student must submit a valid Medical "
                "Certificate from a registered public or private "
                "medical practitioner to the Dean's or "
                "Director's Office within 3 working days, "
                "excluding Saturdays, Sundays and Public "
                "Holidays. Where requirements are satisfied, "
                "Grade N may be awarded and the student may "
                "be allowed to retake the module when next offered."
            ),
            sources["examinations"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/Chap6.pdf"
            )
        ),

        make_record(
            "M3_EXAM_003",
            "University Regulations",
            "Examination Irregularities",
            "Impersonation and Examination Irregularities",
            (
                "Impersonation during examinations is treated "
                "as a serious examination offence. A student "
                "found guilty of impersonation will be expelled "
                "from the University. Other examination "
                "irregularities may be referred to the "
                "Discipline Committee for Examination and "
                "Plagiarism matters."
            ),
            sources["examinations"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/Chap6.pdf"
            )
        ),

        make_record(
            "M3_EXAM_004",
            "University Regulations",
            "Appeals and Reviews",
            "Examination Appeals and Script Reviews",
            (
                "The examination regulations provide an appeal "
                "process for disciplinary decisions arising "
                "from examination irregularities. Students must "
                "follow the prescribed procedure and deadlines. "
                "The regulations also provide procedures for "
                "requesting a review of examination scripts."
            ),
            sources["examinations"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/Chap6.pdf"
            )
        ),


        # =================================================
        # ACADEMIC INTEGRITY
        # =================================================

        make_record(
            "M3_INT_001",
            "University Regulations",
            "Plagiarism",
            "Plagiarism",
            (
                "Plagiarism includes using another person's "
                "work and presenting it as one's own. Examples "
                "include reproducing material without citation, "
                "paraphrasing or summarising without "
                "acknowledgement, using facts, figures, graphs "
                "or information without acknowledgement, and "
                "presenting downloaded Internet material as "
                "one's own."
            ),
            sources["integrity"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap7.pdf"
            )
        ),

        make_record(
            "M3_INT_002",
            "University Regulations",
            "Artificial Intelligence",
            "Use of Artificial Intelligence in Academic Work",
            (
                "The University regulations emphasise "
                "responsible and ethical use of Artificial "
                "Intelligence. AI use must be appropriately "
                "attributed and cited where required. Students "
                "must not misuse AI in ways that compromise "
                "academic integrity, authorship or credibility."
            ),
            sources["integrity"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap7.pdf"
            )
        ),

        make_record(
            "M3_INT_003",
            "University Regulations",
            "Academic Misconduct",
            "Fabrication and Falsification",
            (
                "Fabrication or falsification of results, "
                "data or documents is treated as an academic "
                "offence. Students must not fabricate or "
                "falsify academic results, data or documents."
            ),
            sources["integrity"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap7.pdf"
            )
        ),

        make_record(
            "M3_INT_004",
            "University Regulations",
            "Academic Misconduct",
            "Consequences of Plagiarism",
            (
                "Plagiarism is treated as a serious offence "
                "and may result in disciplinary action. "
                "Depending on the circumstances and applicable "
                "procedures, penalties may include reduction "
                "of academic classification or expulsion. "
                "A qualification may also be forfeited if "
                "plagiarism is detected after an award."
            ),
            sources["integrity"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap7.pdf"
            )
        ),


        # =================================================
        # FINAL YEAR PROJECT / DISSERTATION
        # =================================================

        make_record(
            "M3_FYP_001",
            "University Regulations",
            "Final Year Project",
            "Final Year Project and Dissertation Regulations",
            (
                "Final year project and dissertation regulations "
                "cover project identification and allocation, "
                "project work, dissertation requirements, "
                "referencing, assessment, submission, "
                "re-submission, reviews, declaration forms, "
                "group projects, proposals, progress logs and "
                "non-disclosure agreements where applicable."
            ),
            sources["fyp"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap8.pdf"
            )
        ),

        make_record(
            "M3_FYP_002",
            "University Regulations",
            "Final Year Project",
            "AI Use and Academic Integrity in Projects",
            (
                "Students must ensure project and dissertation "
                "work is their own. Sources, quotations, ideas "
                "and contributions must be acknowledged. "
                "Artificial Intelligence use must be "
                "appropriately attributed and cited or "
                "referenced where applicable."
            ),
            sources["fyp"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap8.pdf"
            )
        ),

        make_record(
            "M3_FYP_003",
            "University Regulations",
            "Final Year Project",
            "Turnitin, Assessment and Late Submission",
            (
                "Final project and dissertation submissions "
                "must comply with the University's Turnitin "
                "requirements. Failure to submit through the "
                "required Turnitin platform may cause the "
                "submission to be treated as unreceivable. "
                "Assessment may include the written dissertation, "
                "viva-voce and/or poster presentation depending "
                "on applicable requirements. Regulations also "
                "provide procedures and penalties for late "
                "submission."
            ),
            sources["fyp"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap8.pdf"
            )
        ),


        # =================================================
        # REGISTRATION AND CONDUCT
        # =================================================

        make_record(
            "M3_CONDUCT_001",
            "University Regulations",
            "Registration and Attendance",
            "Registration, Module Changes and Attendance",
            (
                "University regulations cover student "
                "registration, payment of fees, module "
                "registration, credits, module changes, "
                "module withdrawal, late module registration, "
                "deregistration and module exemption. Students "
                "are expected to attend prescribed lectures, "
                "tutorials and other instruction. Attendance "
                "is monitored and may be considered by relevant "
                "University authorities."
            ),
            sources["conduct"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap3.pdf"
            )
        ),

        make_record(
            "M3_CONDUCT_002",
            "University Regulations",
            "Student Conduct",
            "Student Conduct, Discipline and Withdrawal",
            (
                "Students are expected to conduct themselves "
                "consistently with the objectives, efficiency "
                "and well-being of the University academic "
                "community. Breaches of University discipline "
                "may result in disciplinary action under the "
                "Code of Discipline. Regulations also cover "
                "withdrawal from the University, interruption "
                "of studies, change of programme and official "
                "University notices."
            ),
            sources["conduct"],
            (
                "https://www.uom.ac.mu/images/FILES/"
                "Regulations/2026_2027/chap3.pdf"
            )
        ),


        # =================================================
        # OVERALL REGULATIONS
        # =================================================

        make_record(
            "M3_REG_001",
            "University Regulations",
            "Regulations Overview",
            "University Regulations 2026/2027 Overview",
            (
                "The University publishes regulations covering "
                "admission, programme structure, registration, "
                "assessment, examinations, academic integrity, "
                "final year projects, student welfare, fees and "
                "other academic matters. The supplied 2026/2027 "
                "regulations are marked 'Under Review', so "
                "regulations may be subject to change. "
                "The National Credit Value and Transfer System "
                "(NCVTS) applies to cohorts from Academic Year "
                "2026/2027."
            ),
            sources["regulations"],
            (
                "https://www.uom.ac.mu/index.php/"
                "study-at-uom/current-students/regulations/"
                "undergraduate-postgraduate"
            ),
            academic_year="2026/2027"
        ),
    ]


    return records


# =========================================================
# Main
# =========================================================

def main():

    print(
        "\n"
        + "=" * 65
    )

    print(
        "MEMBER 3 DATA PREPROCESSING"
    )

    print(
        "=" * 65
    )


    validate_source_files()


    records = build_member3_records()


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False
        )


    print(
        "\nMember 3 structured records:",
        len(records)
    )

    print(
        "Output:",
        OUTPUT_FILE
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()