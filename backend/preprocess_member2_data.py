import json
from pathlib import Path


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEMBER2_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "working"
    / "member2"
)

ACADEMIC_FILE = (
    MEMBER2_DATA_DIR
    / "academic-information.txt"
)

STUDENT_SERVICES_FILE = (
    MEMBER2_DATA_DIR
    / "student-services.txt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member2_knowledge.json"
)


# =========================================================
# Validate source files
# =========================================================

def validate_source_files():
    """
    Make sure the Member 2 source files exist
    and are not empty before creating processed records.
    """

    source_files = [
        ACADEMIC_FILE,
        STUDENT_SERVICES_FILE
    ]

    for source_file in source_files:

        if not source_file.exists():
            raise FileNotFoundError(
                f"Source file not found: {source_file}"
            )

        content = source_file.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            raise ValueError(
                f"Source file is empty: {source_file}"
            )


# =========================================================
# Academic information records
# =========================================================

def create_academic_records():
    """
    Convert Member 2 academic information into
    structured retrieval records.
    """

    records = [

        {
            "ID": "ACAD_001",
            "Category": "Academic Information",
            "Subcategory": "Semester 1 Calendar",
            "Title": "Semester 1 Lectures Start",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "Semester 1 lectures start for all students "
                "on Monday 03 August 2026."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_002",
            "Category": "Academic Information",
            "Subcategory": "Semester 1 Calendar",
            "Title": "Semester 1 Module Registration Deadline",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "The deadline for exemption applications, "
                "self-study applications, change of modules, "
                "withdrawal from modules and end of module "
                "registration for Semester 1 is Friday "
                "21 August 2026."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_003",
            "Category": "Academic Information",
            "Subcategory": "Semester 1 Calendar",
            "Title": "Semester 1 Floating Week",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "Floating Week for assessments, lectures, "
                "tutorials and catch-up lectures takes place "
                "from Monday 07 September 2026 to Saturday "
                "12 September 2026."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_004",
            "Category": "Academic Information",
            "Subcategory": "Semester 1 Calendar",
            "Title": "Semester 1 Late Registration Deadline",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "The last date for late module registration "
                "or de-registration upon payment of a penalty "
                "fee is Friday 25 September 2026."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_005",
            "Category": "Academic Information",
            "Subcategory": "Examinations",
            "Title": "Semester 1 Examination Period",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "Semester 1 examinations start on Tuesday "
                "03 November 2026 and end on Saturday "
                "28 November 2026."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_006",
            "Category": "Academic Information",
            "Subcategory": "Semester 2 Calendar",
            "Title": "Semester 2 Induction Period",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "Induction for new entrants takes place from "
                "Monday 11 January 2027 to Friday "
                "15 January 2027."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_007",
            "Category": "Academic Information",
            "Subcategory": "Semester 2 Calendar",
            "Title": "Semester 2 Lectures Start",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "Semester 2 lectures start for all students "
                "on Monday 18 January 2027."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_008",
            "Category": "Academic Information",
            "Subcategory": "Semester 2 Calendar",
            "Title": "Semester 2 Module Registration Deadline",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "The end of module registration for "
                "Semester 2 is Friday 05 February 2027."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_009",
            "Category": "Academic Information",
            "Subcategory": "Semester 2 Calendar",
            "Title": "Students Week",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "Students' Week takes place from Monday "
                "29 March 2027 to Saturday 03 April 2027. "
                "No lectures are held during Students' Week."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_010",
            "Category": "Academic Information",
            "Subcategory": "Semester 2 Calendar",
            "Title": "Semester 2 Revision Weeks",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "Semester 2 Revision Weeks take place from "
                "Monday 12 April 2027 to Saturday "
                "24 April 2027."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_011",
            "Category": "Academic Information",
            "Subcategory": "Examinations",
            "Title": "Semester 2 Examination Period",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "Semester 2 examinations start on Monday "
                "26 April 2027 and end on Saturday "
                "22 May 2027."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "ACAD_012",
            "Category": "Academic Information",
            "Subcategory": "Graduation",
            "Title": "Graduation Ceremonies 2027",
            "Faculty": "University of Mauritius",
            "Academic Year": "2026/2027",
            "Description": (
                "University of Mauritius graduation "
                "ceremonies and sessions are scheduled "
                "during March and April 2027."
            ),
            "Source File": "academic-information.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        }
    ]

    return records


# =========================================================
# Student services records
# =========================================================

def create_student_service_records():
    """
    Convert Member 2 student-services information
    into structured retrieval records.
    """

    records = [

        {
            "ID": "SERV_001",
            "Category": "Student Services",
            "Subcategory": "Library",
            "Title": "Library Opening Hours During Term Time",
            "Faculty": "University of Mauritius",
            "Description": (
                "During term time, the UoM Library is open "
                "Monday to Friday from 08:00 to 17:00 and "
                "Saturday from 08:00 to 13:00."
            ),
            "Source File": "student-services.txt",
            "Source URL": (
                "https://www.uom.ac.mu/index.php/library/"
                "library-regulations/membership"
            ),
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_002",
            "Category": "Student Services",
            "Subcategory": "Library",
            "Title": "Library Opening Hours During Vacation",
            "Faculty": "University of Mauritius",
            "Description": (
                "During the vacation period, the UoM Library "
                "is open Monday to Friday from 08:00 to 16:00 "
                "and Saturday from 08:00 to 12:00."
            ),
            "Source File": "student-services.txt",
            "Source URL": (
                "https://www.uom.ac.mu/index.php/library/"
                "library-regulations/membership"
            ),
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_003",
            "Category": "Student Services",
            "Subcategory": "Library",
            "Title": "Undergraduate Library Borrowing Allowance",
            "Faculty": "University of Mauritius",
            "Description": (
                "Undergraduate students may borrow up to "
                "3 books, including 1 book from the "
                "Reserve Section."
            ),
            "Source File": "student-services.txt",
            "Source URL": (
                "https://www.uom.ac.mu/index.php/library/"
                "library-regulations/membership"
            ),
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_004",
            "Category": "Student Services",
            "Subcategory": "Library",
            "Title": "Academic Staff Library Borrowing Allowance",
            "Faculty": "University of Mauritius",
            "Description": (
                "Academic Staff may borrow up to 6 books, "
                "including 2 books from the Reserve Section."
            ),
            "Source File": "student-services.txt",
            "Source URL": (
                "https://www.uom.ac.mu/index.php/library/"
                "library-regulations/membership"
            ),
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_005",
            "Category": "Student Services",
            "Subcategory": "Library",
            "Title": "MPhil and PhD Library Borrowing Allowance",
            "Faculty": "University of Mauritius",
            "Description": (
                "MPhil and PhD students may borrow up to "
                "5 books, including 1 book from the "
                "Reserve Section."
            ),
            "Source File": "student-services.txt",
            "Source URL": (
                "https://www.uom.ac.mu/index.php/library/"
                "library-regulations/membership"
            ),
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_006",
            "Category": "Student Services",
            "Subcategory": "Library",
            "Title": "Law Student Library Borrowing Allowance",
            "Faculty": "University of Mauritius",
            "Description": (
                "Law students may borrow up to 3 books, "
                "including 1 book from the Reserve Section."
            ),
            "Source File": "student-services.txt",
            "Source URL": (
                "https://www.uom.ac.mu/index.php/library/"
                "library-regulations/membership"
            ),
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_007",
            "Category": "Student Services",
            "Subcategory": "Library",
            "Title": "Non-Academic Staff Library Borrowing Allowance",
            "Faculty": "University of Mauritius",
            "Description": (
                "Non-Academic Staff may borrow up to "
                "1 book."
            ),
            "Source File": "student-services.txt",
            "Source URL": (
                "https://www.uom.ac.mu/index.php/library/"
                "library-regulations/membership"
            ),
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_008",
            "Category": "Student Services",
            "Subcategory": "Computer Labs",
            "Title": "Main CITS Computer Lab Location",
            "Faculty": "University of Mauritius",
            "Description": (
                "The main CITS Computer Lab is located in "
                "the Ex-Student Common Room, First Floor, "
                "VCILT Building."
            ),
            "Source File": "student-services.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_009",
            "Category": "Student Services",
            "Subcategory": "Computer Labs",
            "Title": "CITS Labs 1A 1B and 1C Location",
            "Faculty": "University of Mauritius",
            "Description": (
                "CITS Labs 1A, 1B and 1C are located in "
                "the Phase II Engineering Building."
            ),
            "Source File": "student-services.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_010",
            "Category": "Student Services",
            "Subcategory": "Computer Labs",
            "Title": "Other CITS and Faculty Computer Labs",
            "Faculty": "University of Mauritius",
            "Description": (
                "The CITS FSSH Lab is on the Ground Floor "
                "of the Faculty of Social Sciences and "
                "Humanities Building. CITS FOA Lab 1A is "
                "on the First Floor of the Faculty of "
                "Agriculture Building. CITS FOA Lab 2A is "
                "on the Second Floor of the Faculty of "
                "Agriculture Building. CITS ETB Lab and "
                "the Bioinformatics Lab are on the First "
                "Floor of the Prof E. Lim Fat Building. "
                "The CITS Ebene Core Lab is on the second "
                "floor of The Core Building, Ebene."
            ),
            "Source File": "student-services.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_011",
            "Category": "Student Services",
            "Subcategory": "Payments",
            "Title": "UoM Online Credit Card Payment",
            "Faculty": "University of Mauritius",
            "Description": (
                "University fees may be paid through the "
                "UoM Pay Online portal using a credit card "
                "in Mauritian Rupees or US Dollars."
            ),
            "Source File": "student-services.txt",
            "Source URL": (
                "https://apply.uom.ac.mu/UOMonlinePay/"
                "uomonlinepay.aspx"
            ),
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_012",
            "Category": "Student Services",
            "Subcategory": "Payments",
            "Title": "MCB Direct Transfer for UoM Payments",
            "Faculty": "University of Mauritius",
            "Description": (
                "Payments may be made to the Mauritius "
                "Commercial Bank UoM Bank Account. "
                "The account number is 000142555444. "
                "Payments may be made through MCB Internet "
                "Banking and the MCB JUICE mobile app."
            ),
            "Source File": "student-services.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_013",
            "Category": "Student Services",
            "Subcategory": "Payments",
            "Title": "SBM Mauritian Rupee Account for UoM Payments",
            "Faculty": "University of Mauritius",
            "Description": (
                "The SBM Bank Mauritius account for "
                "payments in Mauritian Rupees has account "
                "number 62025100002399 and may be accessed "
                "through SBM Internet Banking."
            ),
            "Source File": "student-services.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_014",
            "Category": "Student Services",
            "Subcategory": "Payments",
            "Title": "SBM US Dollar Account for UoM Payments",
            "Faculty": "University of Mauritius",
            "Description": (
                "The SBM Bank Mauritius account for "
                "payments in US Dollars has account number "
                "61026000014607."
            ),
            "Source File": "student-services.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        },

        {
            "ID": "SERV_015",
            "Category": "Student Services",
            "Subcategory": "Payments",
            "Title": "UoM Digital and Mobile Payment Applications",
            "Faculty": "University of Mauritius",
            "Description": (
                "Digital and mobile payment applications "
                "accepted by UoM include my.t Billpay "
                "through the my.t money mobile application "
                "and the BLINK mobile payment application."
            ),
            "Source File": "student-services.txt",
            "Source URL": "",
            "Verified Status": "Verified"
        }
    ]

    return records


# =========================================================
# Main preprocessing function
# =========================================================

def main():

    print(
        "Validating Member 2 source files..."
    )

    validate_source_files()


    academic_records = (
        create_academic_records()
    )

    service_records = (
        create_student_service_records()
    )

    combined_records = (
        academic_records
        + service_records
    )


    # -----------------------------------------------------
    # Check for duplicate IDs
    # -----------------------------------------------------

    record_ids = [
        record["ID"]
        for record in combined_records
    ]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "Duplicate record IDs detected."
        )


    # -----------------------------------------------------
    # Create processed directory if required
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            combined_records,
            file,
            indent=4,
            ensure_ascii=False
        )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(
        f"Academic records: "
        f"{len(academic_records)}"
    )

    print(
        f"Student service records: "
        f"{len(service_records)}"
    )

    print(
        f"Total Member 2 records: "
        f"{len(combined_records)}"
    )

    print(
        "Processed data saved to:"
    )

    print(
        OUTPUT_FILE
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()