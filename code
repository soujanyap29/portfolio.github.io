#include <iostream>
#include <string>
#include <unordered_map>
#include <list>
#include <vector>
#include <filesystem>
#include <fstream>
#include <ctime>
#include <regex>
#include <conio.h>
#include <direct.h>
#include <set>
#include <numeric>
#include <map>
#include <queue>
#include <windows.h>  // For Sleep function
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
namespace fs = std::filesystem;

class Logger {
public:
    static void init() {
        // Initialize logging system
        std::cout << "Logging system initialized\n";
    }

    static void log(const std::string& message) {
        std::cout << "[LOG] " << message << std::endl;
    }
};


class PasswordValidator {
public:
    static void displayStatus(const std::string& password) {
        std::cout << "\rPassword length: " << password.length() << " characters";
    }
};

class PersonalInfo {
private:
    std::string dateOfBirth;
    std::string bloodGroup;
    std::string phone;
    std::string email;
    std::string address;
    int age;           // Add here
    double height;     // Add here
    double weight;

    struct EmergencyContact {
        std::string name;
        std::string relationship;
        std::string phone;
    } emergency;

    struct Insurance {
        std::string provider;
        std::string policyNumber;
        std::string expiryDate;
    } insurance;

public:
    // Setters
    void setDateOfBirth(const std::string& dob) {
        dateOfBirth = dob;
    }

    void setBloodGroup(const std::string& bg) {
        static const std::vector<std::string> validGroups =
            {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"};
        if(std::find(validGroups.begin(), validGroups.end(), bg) != validGroups.end()) {
            bloodGroup = bg;
        }
    }

    void setContactInfo(const std::string& ph, const std::string& em, const std::string& addr) {
        phone = ph;
        email = em;
        address = addr;
    }

    void setEmergencyContact(const std::string& name, const std::string& rel, const std::string& phone) {
        emergency.name = name;
        emergency.relationship = rel;
        emergency.phone = phone;
    }

    void setInsurance(const std::string& provider, const std::string& policy, const std::string& expiry) {
        insurance.provider = provider;
        insurance.policyNumber = policy;
        insurance.expiryDate = expiry;
    }

    // Getters
    std::string getDateOfBirth() const {
        return dateOfBirth;
    }

    std::string getBloodGroup() const {
        return bloodGroup;
    }

    std::string getPhone() const {
        return phone;
    }

    std::string getEmail() const {
        return email;
    }

    std::string getAddress() const {
        return address;
    }

void setHeight(double h) { height = h; }
    void setWeight(double w) { weight = w; }
    void setAge(int a) { age = a; }

    double getHeight() const { return height; }
    double getWeight() const { return weight; }
    int getAge() const { return age; }

    std::string getEmergencyName() const {
        return emergency.name;
    }

    std::string getEmergencyRelationship() const {
        return emergency.relationship;
    }

    std::string getEmergencyPhone() const {
        return emergency.phone;
    }

    std::string getInsuranceProvider() const {
        return insurance.provider;
    }

    std::string getInsurancePolicyNumber() const {
        return insurance.policyNumber;
    }

    std::string getInsuranceExpiryDate() const {
        return insurance.expiryDate;
    }
};

struct FileInfo {
    std::filesystem::path filePath;
    std::time_t timestamp;

    FileInfo(const std::filesystem::path& path, std::time_t ts)
        : filePath(path), timestamp(ts) {}

    bool operator<(const FileInfo& other) const {
        return timestamp > other.timestamp;
    }
};


class RecordManager {
private:
    std::map<std::string, std::vector<std::string>> fileRecords;
    std::priority_queue<FileInfo> deleteQueue;
public:
    void createDirectories(const std::string& patientId);
    bool fileExists(const std::string& filePath);
    void copyBinaryFile(const std::string& sourcePath, const std::string& destinationDir);
    void retrieveFile(const std::string& idType, const std::string& id);
    void addToDeleteQueue(const std::string& idType, const std::string& id, const std::string& fileName);
    void deleteFileFromQueue();
    void listFiles(const std::string& idType, const std::string& id);
    void showFileRecords(const std::string& id);
    void handleFileStorage() {
        std::string idType, id, filePath;

        std::cout << "Enter ID type (patient/nurse/caregiver): ";
        std::getline(std::cin, idType);
        std::transform(idType.begin(), idType.end(), idType.begin(), ::toupper);

        std::cout << "Enter ID: ";
        std::getline(std::cin, id);

        std::string folderPath = "C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\PATH\\" + idType + "_" + id;
        if (std::filesystem::exists(folderPath)) {
            std::cout << "\nFolder Status: EXISTS at " << folderPath << "\n";
        } else {
            std::cout << "\nFolder Status: NOT FOUND - Creating new folder...\n";
            this->createDirectories(idType + "_" + id);
        }

        std::cout << "\nEnter file path: ";
        std::getline(std::cin, filePath);

        this->copyBinaryFile(filePath, idType + "_" + id);
    }
};


class MedicalRecord {
private:
    std::string recordID;
    std::string patientID;
    std::string diagnosis;
    std::string prescription;
    std::string doctorName;
    std::string date;
    double billAmount;

public:
    MedicalRecord(const std::string& pid, const std::string& diag,
                 const std::string& pres, const std::string& doc, double bill)
        : patientID(pid), diagnosis(diag), prescription(pres),
          doctorName(doc), billAmount(bill) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        date = std::ctime(&now_time);
        recordID = "REC" + std::to_string(std::hash<std::string>{}(date + pid));
    }

    std::string getRecordID() const { return recordID; }
    double getBillAmount() const { return billAmount; }
    std::string getDate() const { return date; }
};



class UserProfile {
private:
    std::string userID;
    std::string name;
    std::string password;
    std::vector<std::string> medicalFiles;
    static std::unordered_map<std::string, UserProfile*> userDatabase;
    static int currentID;
    PersonalInfo personalInfo;
     // Add this line at the top with other private members

    std::string generateUniqueID() {
    std::string newID;
    do {
        currentID++;
        std::string idNumber = std::to_string(currentID);
        std::string paddedID = std::string(4 - idNumber.length(), '0') + idNumber;
        newID = "USER" + paddedID;
    } while (userDatabase.find(newID) != userDatabase.end());

    return newID;
}


    bool isValidName(const std::string& name) {
        return std::regex_match(name, std::regex("^[A-Za-z ]+$"));
    }

    bool isStrongPassword(const std::string& pwd) {
        bool hasUpper = false, hasLower = false, hasDigit = false, hasSpecial = false;
        if(pwd.length() < 8) return false;

        for(char c : pwd) {
            if(isupper(c)) hasUpper = true;
            else if(islower(c)) hasLower = true;
            else if(isdigit(c)) hasDigit = true;
            else hasSpecial = true;
        }

        return hasUpper && hasLower && hasDigit && hasSpecial;
    }

    bool isValidFileFormat(const std::string& filename) {
        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        return (ext == "pdf" || ext == "jpg" || ext == "png");
    }

    void createUserDirectory() {
        std::string path = "C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\PATH\\" + userID;
        _mkdir(path.c_str());
    }

static void initializeCurrentID() {
        std::ifstream file("C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\file.txt");
        std::string line;
        std::set<int> usedIDs;

        while (getline(file, line)) {
            if (line.find("ID: USER") != std::string::npos) {
                int id = std::stoi(line.substr(9));
                usedIDs.insert(id);
            }
        }

        currentID = usedIDs.empty() ? 0 : *usedIDs.rbegin();
    }

public:
    virtual ~UserProfile() {}
    static void initialize() {
        initializeCurrentID();
    }
    static std::vector<UserProfile*> searchUsers(const std::string& searchTerm) {
        std::vector<UserProfile*> results;
        for (const auto& [id, user] : userDatabase) {
            if (user->getName().find(searchTerm) != std::string::npos ||
                id.find(searchTerm) != std::string::npos) {
                results.push_back(user);
            }
        }
        return results;
    }
    void updatePersonalInfo(const PersonalInfo& info) {
        personalInfo = info;
    }

    const PersonalInfo& getPersonalInfo() const {
        return personalInfo;
    }

    UserProfile(const std::string& n, const std::string& pwd) {
        if(!isValidName(n)) {
            throw std::invalid_argument("Invalid name format");
        }

        if(!isStrongPassword(pwd)) {
            throw std::invalid_argument("Password not strong enough");
        }

        name = n;
        password = pwd;
        userID = generateUniqueID();
        createUserDirectory();
        userDatabase[userID] = this;
    }

    bool uploadFile(const std::string& filepath) {
        if(!isValidFileFormat(filepath)) {
            return false;
        }

        std::string filename = filepath.substr(filepath.find_last_of("\\") + 1);
        std::string destPath = "C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\PATH\\" +
                              userID + "\\" + filename;

        for(const auto& file : medicalFiles) {
            if(file == filename) return false;
        }

        try {
            fs::copy_file(filepath, destPath, fs::copy_options::overwrite_existing);
            medicalFiles.push_back(filename);
            return true;
        } catch(...) {
            return false;
        }
    }

    std::string getUserID() const { return userID; }
    std::string getName() const { return name; }

    static UserProfile* getUser(const std::string& id) {
        auto it = userDatabase.find(id);
        return (it != userDatabase.end()) ? it->second : nullptr;
    }

    static const std::unordered_map<std::string, UserProfile*>& getUserDatabase() {
        return userDatabase;
    }


};

std::unordered_map<std::string, UserProfile*> UserProfile::userDatabase;
int UserProfile::currentID = 0;

class NurseProfile : public UserProfile {
private:
    // Professional Identifiers
    std::string nurseID;
    std::string licenseNumber;
    std::string department;

    // Professional Details
    std::string specialty;
    int yearsOfExperience;
    std::string certifications;
    std::vector<int> ratings;
    bool availability;
    std::vector<std::string> reviews;

    // Static Members
    static std::unordered_map<std::string, NurseProfile*> nurseDatabase;
    static int currentNurseID;

    std::string generateNurseID() {
        currentNurseID++;
        return "NURSE" + std::to_string(currentNurseID);
    }

public:
    ~NurseProfile() override {}

    NurseProfile(const std::string& name, const std::string& pwd,
                const std::string& license, const std::string& dept,
                const std::string& spec, int experience, const std::string& certs)
        : UserProfile(name, pwd) {
        nurseID = generateNurseID();
        licenseNumber = license;
        department = dept;
        specialty = spec;
        yearsOfExperience = experience;
        certifications = certs;
        availability = true;
        nurseDatabase[nurseID] = this;
    }

    // Professional Info Getters
    std::string getNurseID() const { return nurseID; }
    std::string getLicenseNumber() const { return licenseNumber; }
    std::string getDepartment() const { return department; }
    std::string getSpecialty() const { return specialty; }
    int getExperience() const { return yearsOfExperience; }
    std::string getCertifications() const { return certifications; }
    bool isAvailable() const { return availability; }

    // Management Methods
    void setAvailability(bool status) { availability = status; }
    void addRating(int rating) { ratings.push_back(rating); }
    void addReview(const std::string& review) { reviews.push_back(review); }
    const std::vector<int>& getRatings() const { return ratings; }
    const std::vector<std::string>& getReviews() const { return reviews; }

    double calculateAverageRating() const {
        if (ratings.empty()) return 0.0;
        return static_cast<double>(std::accumulate(ratings.begin(), ratings.end(), 0)) / ratings.size();
    }

    static NurseProfile* getNurse(const std::string& id) {
        auto it = nurseDatabase.find(id);
        return (it != nurseDatabase.end()) ? it->second : nullptr;
    }
};




void displayPasswordGuidelines() {
    std::cout << "\n=== Password Requirements ===\n"
              << "✓ Minimum 8 characters long\n"
              << "✓ At least 1 uppercase letter (A-Z)\n"
              << "✓ At least 1 lowercase letter (a-z)\n"
              << "✓ At least 1 number (0-9)\n"
              << "✓ At least 1 special character (!@#$%^&*()_+-=[]{}|;:,.<>?)\n"
              << "\nExample of a strong password: P@ssw0rd_2023\n"
              << "----------------------------------------\n";
}

class NurseManagement {
private:
    struct Nurse {
        std::string id;
        std::string name;
        std::string department;
        std::string specialty;
        int experience;
        bool available;
    };

    struct NurseRating {
        double professionalism;
        double communication;
        double technicalExpertise;
        double empathy;
        double punctuality;
        std::string comment;
        std::string date;

        double getAverageRating() const {
            return (professionalism + communication + technicalExpertise + empathy + punctuality) / 5.0;
        }
    };

    static std::map<std::string, Nurse> nurses;
    static std::map<std::string, std::vector<NurseRating>> nurseRatings;

    static std::string getCurrentDateTime() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    static void saveNurseData() {
        std::ofstream file("C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\nursefile.txt");
        for (const auto& [id, ratings] : nurseRatings) {
            file << "NURSE_ID:" << id << "\n";
            for (const auto& rating : ratings) {
                file << rating.professionalism << "|"
                     << rating.communication << "|"
                     << rating.technicalExpertise << "|"
                     << rating.empathy << "|"
                     << rating.punctuality << "|"
                     << rating.date << "|"
                     << rating.comment << "\n";
            }
            file << "END_NURSE\n";
        }
        file.close();
    }

static void displaySortedList(const vector<NurseProfile*>& nurses, const string& sortType) {
        cout << "\nNurses Sorted by " << sortType << ":\n";
        cout << "==========================================\n";
        for (const auto& nurse : nurses) {
            cout << "ID: " << nurse->getNurseID()
                 << " | Name: " << nurse->getName()
                 << " | Experience: " << nurse->getExperience() << " years"
                 << " | Rating: " << nurse->calculateAverageRating() << "/5\n";
        }
    }

    static void searchByDepartment(const string& dept) {
        bool found = false;
        for (const auto& pair : NurseProfile::nurseDatabase) {
            if (pair.second->getDepartment() == dept) {
                found = true;
                cout << "\nID: " << pair.second->getNurseID()
                     << "\nName: " << pair.second->getName()
                     << "\nExperience: " << pair.second->getExperience() << " years\n";
            }
        }
        if (!found) cout << "No nurses found in this department.\n";
    }

public:
    static void sortByName() {
        vector<NurseProfile*> nurseList;
        for (const auto& pair : NurseProfile::nurseDatabase) {
            nurseList.push_back(pair.second);
        }

        sort(nurseList.begin(), nurseList.end(),
             [](NurseProfile* a, NurseProfile* b) {
                 return a->getName() < b->getName();
             });

        displaySortedList(nurseList, "Name");
    }

    static void sortByExperience() {
        vector<NurseProfile*> nurseList;
        for (const auto& pair : NurseProfile::nurseDatabase) {
            nurseList.push_back(pair.second);
        }

        sort(nurseList.begin(), nurseList.end(),
             [](NurseProfile* a, NurseProfile* b) {
                 return a->getExperience() > b->getExperience();
             });

        displaySortedList(nurseList, "Experience");
    }

    static void sortByRating() {
        vector<NurseProfile*> nurseList;
        for (const auto& pair : NurseProfile::nurseDatabase) {
            nurseList.push_back(pair.second);
        }

        sort(nurseList.begin(), nurseList.end(),
             [](NurseProfile* a, NurseProfile* b) {
                 return a->calculateAverageRating() > b->calculateAverageRating();
             });

        displaySortedList(nurseList, "Rating");
    }

    static void advancedSearch() {
        cout << "\nAdvanced Search Options:\n"
             << "1. Search by Department\n"
             << "2. Search by Experience Range\n"
             << "3. Search by Rating Range\n"
             << "Enter choice: ";

        string choice;
        getline(cin, choice);

        if (choice == "1") {
            string dept;
            cout << "Enter department: ";
            getline(cin, dept);
            searchByDepartment(dept);
        }
        // Add other search options as needed
    }
    static void searchNurseById() {
        string targetId;
        cout << "Enter Nurse ID to search: ";
        getline(cin, targetId);

        vector<NurseProfile*> nurseList;
        for (const auto& pair : NurseProfile::nurseDatabase) {
            nurseList.push_back(pair.second);
        }

        int index = binarySearchNurse(nurseList, targetId);
        if (index != -1) {
            cout << "\nNurse found!\n";
            cout << "ID: " << nurseList[index]->getNurseID() << "\n";
            cout << "Name: " << nurseList[index]->getName() << "\n";
            cout << "Department: " << nurseList[index]->getDepartment() << "\n";
            cout << "Specialty: " << nurseList[index]->getSpecialty() << "\n";
        } else {
            cout << "Nurse not found.\n";
        }
    }

    static void sortAndDisplayNurses() {
        vector<NurseProfile*> nurseList;
        for (const auto& pair : NurseProfile::nurseDatabase) {
            nurseList.push_back(pair.second);
        }

        heapSort(nurseList);

        cout << "\nSorted Nurses List:\n";
        cout << "==================\n";
        for (const auto& nurse : nurseList) {
            cout << "ID: " << nurse->getNurseID()
                 << " | Name: " << nurse->getName()
                 << " | Department: " << nurse->getDepartment() << "\n";
        }
    }
        static void loadNurseData() {
    std::ifstream file("C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\nursefile.txt");
    std::string line;
    std::string currentId;

    while (std::getline(file, line)) {
        try {
            if (line.substr(0, 9) == "NURSE_ID:") {
                currentId = line.substr(9);
                Nurse nurse;
                nurse.id = currentId;
                nurses[currentId] = nurse;  // Add this line to populate nurses map
            } else if (line != "END_NURSE" && !line.empty()) {
                std::stringstream ss(line);
                std::string item;
                NurseRating rating;

                if (std::getline(ss, item, '|')) rating.professionalism = std::stod(item);
                if (std::getline(ss, item, '|')) rating.communication = std::stod(item);
                if (std::getline(ss, item, '|')) rating.technicalExpertise = std::stod(item);
                if (std::getline(ss, item, '|')) rating.empathy = std::stod(item);
                if (std::getline(ss, item, '|')) rating.punctuality = std::stod(item);
                if (std::getline(ss, item, '|')) rating.date = item;
                if (std::getline(ss, rating.comment)) {
                    nurseRatings[currentId].push_back(rating);
                }
            }
        } catch (const std::exception& e) {
            continue; // Skip invalid entries and continue loading
        }
    }
    file.close();
}

    static void addNurse() {
        Nurse nurse;
        std::cout << "Enter Nurse ID: ";
        std::getline(std::cin, nurse.id);
        std::cout << "Enter Name: ";
        std::getline(std::cin, nurse.name);
        std::cout << "Enter Department: ";
        std::getline(std::cin, nurse.department);
        std::cout << "Enter Specialty: ";
        std::getline(std::cin, nurse.specialty);
        std::cout << "Enter Experience (years): ";
        std::cin >> nurse.experience;
        nurse.available = true;

        nurses[nurse.id] = nurse;
        saveNurseData();
        std::cout << "Nurse added successfully!\n";
    }

    static void viewNurses() {
        std::cout << "\n=== Registered Nurses ===\n";
        for (const auto& [id, nurse] : nurses) {
            std::cout << "\nID: " << nurse.id
                     << "\nName: " << nurse.name
                     << "\nDepartment: " << nurse.department
                     << "\nSpecialty: " << nurse.specialty
                     << "\nExperience: " << nurse.experience << " years"
                     << "\nAvailability: " << (nurse.available ? "Available" : "Not Available")
                     << "\nAverage Rating: " << calculateAverageRating(id) << "/5\n"
                     << "----------------------------------------\n";
        }
    }

    static void rateNurse() {
        std::string nurseId;
        std::cout << "Enter Nurse ID: ";
        std::getline(std::cin, nurseId);

        if (nurses.find(nurseId) == nurses.end()) {
            std::cout << "Nurse not found!\n";
            return;
        }

        NurseRating rating;
        std::cout << "\nPlease rate the following aspects (1-5):\n\n";

        std::cout << "Professionalism: ";
        std::cin >> rating.professionalism;
        std::cout << "Communication: ";
        std::cin >> rating.communication;
        std::cout << "Technical Expertise: ";
        std::cin >> rating.technicalExpertise;
        std::cout << "Empathy: ";
        std::cin >> rating.empathy;
        std::cout << "Punctuality: ";
        std::cin >> rating.punctuality;

        std::cin.ignore();
        std::cout << "Additional Comments: ";
        std::getline(std::cin, rating.comment);

        rating.date = getCurrentDateTime();
        nurseRatings[nurseId].push_back(rating);
        saveNurseData();

        std::cout << "Rating submitted successfully!\n";
    }

    static void viewRatings() {
        std::string nurseId;
        std::cout << "Enter Nurse ID: ";
        std::getline(std::cin, nurseId);

        if (nurses.find(nurseId) == nurses.end()) {
            std::cout << "Nurse not found!\n";
            return;
        }

        const auto& ratings = nurseRatings[nurseId];
        if (ratings.empty()) {
            std::cout << "No ratings available for " << nurses[nurseId].name << "\n";
            return;
        }

        std::cout << "\n=== Rating History for " << nurses[nurseId].name << " ===\n";
        for (const auto& rating : ratings) {
            std::cout << "\nDate: " << rating.date
                     << "\nProfessionalism: " << rating.professionalism << "/5"
                     << "\nCommunication: " << rating.communication << "/5"
                     << "\nTechnical Expertise: " << rating.technicalExpertise << "/5"
                     << "\nEmpathy: " << rating.empathy << "/5"
                     << "\nPunctuality: " << rating.punctuality << "/5"
                     << "\nOverall Rating: " << rating.getAverageRating() << "/5"
                     << "\nComments: " << rating.comment
                     << "\n----------------------------------------\n";
        }
    }

    static void searchNurse() {
        std::string searchTerm;
        std::cout << "Enter name, department, or specialty to search: ";
        std::getline(std::cin, searchTerm);

        std::transform(searchTerm.begin(), searchTerm.end(), searchTerm.begin(), ::tolower);

        bool found = false;
        for (const auto& [id, nurse] : nurses) {
            std::string name = nurse.name;
            std::string dept = nurse.department;
            std::string spec = nurse.specialty;

            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            std::transform(dept.begin(), dept.end(), dept.begin(), ::tolower);
            std::transform(spec.begin(), spec.end(), spec.begin(), ::tolower);

            if (name.find(searchTerm) != std::string::npos ||
                dept.find(searchTerm) != std::string::npos ||
                spec.find(searchTerm) != std::string::npos) {
                found = true;
                std::cout << "\nID: " << id
                         << "\nName: " << nurse.name
                         << "\nDepartment: " << nurse.department
                         << "\nSpecialty: " << nurse.specialty
                         << "\nRating: " << calculateAverageRating(id) << "/5\n";
            }
        }

        if (!found) {
            std::cout << "No matches found.\n";
        }
    }

    static void topRatedNurse() {
        if (nurseRatings.empty()) {
            std::cout << "No ratings available in the system.\n";
            return;
        }

        std::string topNurseId;
        double highestRating = 0.0;

        for (const auto& [id, ratings] : nurseRatings) {
            double avgRating = calculateAverageRating(id);
            if (avgRating > highestRating) {
                highestRating = avgRating;
                topNurseId = id;
            }
        }

        if (!topNurseId.empty()) {
            std::cout << "\n=== Top Rated Nurse ===\n"
                     << "Name: " << nurses[topNurseId].name
                     << "\nDepartment: " << nurses[topNurseId].department
                     << "\nSpecialty: " << nurses[topNurseId].specialty
                     << "\nOverall Rating: " << highestRating << "/5\n";
        }
    }

    static void generateDetailedReport() {
        std::ofstream report("nurse_performance_report.txt");
        report << "=== Nurse Performance Report ===\n";
        report << "Generated on: " << getCurrentDateTime() << "\n\n";

        for (const auto& [id, nurse] : nurses) {
            const auto& ratings = nurseRatings[id];
            double avgProf = 0, avgComm = 0, avgTech = 0, avgEmp = 0, avgPunct = 0;

            for (const auto& rating : ratings) {
                avgProf += rating.professionalism;
                avgComm += rating.communication;
                avgTech += rating.technicalExpertise;
                avgEmp += rating.empathy;
                avgPunct += rating.punctuality;
            }

            int numRatings = ratings.size();
            if (numRatings > 0) {
                avgProf /= numRatings;
                avgComm /= numRatings;
                avgTech /= numRatings;
                avgEmp /= numRatings;
                avgPunct /= numRatings;
            }

            report << "Nurse ID: " << id
                  << "\nName: " << nurse.name
                  << "\nDepartment: " << nurse.department
                  << "\nSpecialty: " << nurse.specialty
                  << "\n\nPerformance Metrics:"
                  << "\nProfessionalism: " << avgProf << "/5"
                  << "\nCommunication: " << avgComm << "/5"
                  << "\nTechnical Expertise: " << avgTech << "/5"
                  << "\nEmpathy: " << avgEmp << "/5"
                  << "\nPunctuality: " << avgPunct << "/5"
                  << "\n\nTotal Reviews: " << numRatings
                  << "\nOverall Rating: " << calculateAverageRating(id) << "/5"
                  << "\n----------------------------------------\n\n";
        }

        report.close();
        std::cout << "Detailed report generated successfully: nurse_performance_report.txt\n";
    }

    static double calculateAverageRating(const std::string& nurseId) {
        const auto& ratings = nurseRatings[nurseId];
        if (ratings.empty()) return 0.0;

        double sum = 0.0;
        for (const auto& rating : ratings) {
            sum += rating.getAverageRating();
        }
        return sum / ratings.size();
    }
};

// Initialize static members
std::map<std::string, NurseManagement::Nurse> NurseManagement::nurses;
std::map<std::string, std::vector<NurseManagement::NurseRating>> NurseManagement::nurseRatings;



class UserInterface {
public:
    static std::string getHiddenPassword() {
        std::string password;
        char ch;
        std::cout << "Enter password: \n";
        while ((ch = _getch()) != '\r') {
            if (ch == '\b') {
                if (!password.empty()) {
                    password.pop_back();
                }
            } else {
                password += ch;
            }
            PasswordValidator::displayStatus(password);
        }
        std::cout << "\n\n";
        return password;
    }

    static bool isValidDate(const std::string& date) {
        if (date.length() != 10) return false;
        if (date[2] != '/' || date[5] != '/') return false;

        int day = std::stoi(date.substr(0, 2));
        int month = std::stoi(date.substr(3, 2));
        int year = std::stoi(date.substr(6, 4));

        if (month < 1 || month > 12) return false;
        if (year < 1900 || year > 2023) return false;

        int daysInMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        if (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))
            daysInMonth[1] = 29;

        return day > 0 && day <= daysInMonth[month - 1];
    }

    static void searchUser() {
        std::string searchTerm;
        std::cout << "Enter name or ID to search: ";
        std::getline(std::cin, searchTerm);

        std::vector<UserProfile*> results = UserProfile::searchUsers(searchTerm);

        if (results.empty()) {
            std::cout << "No matching users found.\n";
            return;
        }

        std::cout << "\nFound " << results.size() << " matching users:\n";
        for (const auto& user : results) {
            displayUserInfo(user, user->getPersonalInfo());
        }
    }

    static void exportUserData(const std::string& userID) {
        UserProfile* user = UserProfile::getUser(userID);
        if (!user) {
            std::cout << "User not found!\n";
            return;
        }

        std::string exportPath = "exports/" + userID + "_profile.txt";
        std::filesystem::create_directories("exports");

        std::ofstream file(exportPath);
        file << "=== User Profile ===\n";
        file << "ID: " << user->getUserID() << "\n";
        file << "Name: " << user->getName() << "\n";
        const PersonalInfo& info = user->getPersonalInfo();
        file << "Date of Birth: " << info.getDateOfBirth() << "\n";
        file << "Blood Group: " << info.getBloodGroup() << "\n";
        file << "Phone: " << info.getPhone() << "\n";
        file << "Email: " << info.getEmail() << "\n";
        file << "Address: " << info.getAddress() << "\n";
        file.close();

        std::cout << "User data exported successfully to " << exportPath << "\n";
    }

    static void registerUser() {
        std::string name, password;
        std::cout << "Enter name: ";
        std::getline(std::cin, name);

        displayPasswordGuidelines();
        password = getHiddenPassword();

        try {
            UserProfile* user = new UserProfile(name, password);
            std::cout << "User registered successfully! ID: " << user->getUserID() << std::endl;
            collectPersonalInfo(user);
        } catch(const std::exception& e) {
            std::cout << "Registration failed: " << e.what() << std::endl;
        }
    }

    static void registerNurse() {
        std::string name, password, license, department, specialty;
        int experience;
        std::string certifications;

        std::cout << "Enter nurse name: ";
        std::getline(std::cin, name);

        std::cout << "Enter license number: ";
        std::getline(std::cin, license);

        std::cout << "Enter department: ";
        std::getline(std::cin, department);

        std::cout << "Enter specialty: ";
        std::getline(std::cin, specialty);

        std::cout << "Enter years of experience: ";
        std::cin >> experience;
        std::cin.ignore();

        std::cout << "Enter certifications: ";
        std::getline(std::cin, certifications);

        displayPasswordGuidelines();
        password = getHiddenPassword();

        try {
            NurseProfile* nurse = new NurseProfile(name, password, license, department,
                                                 specialty, experience, certifications);
            std::cout << "Nurse registered successfully!\n";
            std::cout << "Nurse ID: " << nurse->getNurseID() << "\n";
            collectPersonalInfo(nurse);
        } catch(const std::exception& e) {
            std::cout << "Registration failed: " << e.what() << "\n";
        }
    }

    static void uploadUserFile() {
        std::string userID;
        std::cout << "Enter user ID: ";
        std::getline(std::cin, userID);

        UserProfile* user = UserProfile::getUser(userID);
        if (!user) {
            std::cout << "User not found!" << std::endl;
            return;
        }

        std::cout << "Enter file path: ";
        std::string filepath;
        std::getline(std::cin, filepath);

        if (user->uploadFile(filepath)) {
            std::cout << "File uploaded successfully!" << std::endl;
        } else {
            std::cout << "File upload failed!" << std::endl;
        }
    }

    static void collectPersonalInfo(UserProfile* user) {
        std::cout << "\nPlease enter your password to continue: ";
    std::string password = getHiddenPassword();

    // Here you would verify the password against stored password
    // For demonstration, we'll use a simple check
    if (password != "password123") {  // Replace with actual password verification
        std::cout << "Access denied. Invalid password.\n";
        return;
    }
    PersonalInfo info = user->getPersonalInfo();
    std::string input;

    std::cout << "\n=== Personal Information Collection ===\n";

    // Date of Birth
    do {
        std::cout << "Enter Date of Birth (DD/MM/YYYY): ";
        std::getline(std::cin, input);
    } while (!isValidDate(input));
    info.setDateOfBirth(input);
    info.setAge(calculateAge(input));

    // Blood Group
    do {
        std::cout << "Enter Blood Group (A+/A-/B+/B-/AB+/AB-/O+/O-): ";
        std::getline(std::cin, input);
    } while (input != "A+" && input != "A-" && input != "B+" && input != "B-" &&
             input != "AB+" && input != "AB-" && input != "O+" && input != "O-");
    info.setBloodGroup(input);

    // Contact Information
    std::string phone, email, address;
    do {
        std::cout << "Enter Phone Number (10 digits): ";
        std::getline(std::cin, phone);
    } while (!isValidPhoneNumber(phone));

    do {
        std::cout << "Enter Email: ";
        std::getline(std::cin, email);
    } while (!isValidEmail(email));

    std::cout << "Enter Address: ";
    std::getline(std::cin, address);

    info.setContactInfo(phone, email, address);

    // Height and Weight
    double height, weight;
    std::cout << "Enter Height (cm): ";
    std::cin >> height;
    std::cin.ignore();
    info.setHeight(height);

    std::cout << "Enter Weight (kg): ";
    std::cin >> weight;
    std::cin.ignore();
    info.setWeight(weight);

    // Emergency Contact
    std::string emergencyName, relationship, emergencyPhone;
    std::cout << "\n=== Emergency Contact Information ===\n";
    std::cout << "Enter Name: ";
    std::getline(std::cin, emergencyName);
    std::cout << "Enter Relationship: ";
    std::getline(std::cin, relationship);
    do {
        std::cout << "Enter Phone Number (10 digits): ";
        std::getline(std::cin, emergencyPhone);
    } while (!isValidPhoneNumber(emergencyPhone));

    info.setEmergencyContact(emergencyName, relationship, emergencyPhone);

    // Insurance Information
    std::string provider, policyNumber, expiryDate;
    std::cout << "\n=== Insurance Information ===\n";
    do {
        std::cout << "Enter Insurance Provider: ";
        std::getline(std::cin, provider);
    } while (!isValidProvider(provider));

    do {
        std::cout << "Enter Policy Number (10 digits): ";
        std::getline(std::cin, policyNumber);
    } while (!isValidPolicyNumber(policyNumber));

    do {
        std::cout << "Enter Expiry Date (DD/MM/YYYY): ";
        std::getline(std::cin, expiryDate);
    } while (!isValidDate(expiryDate));

    info.setInsurance(provider, policyNumber, expiryDate);

    user->updatePersonalInfo(info);
    std::cout << "\nPersonal information updated successfully!\n";
}


private:
    static void displayUserInfo(const UserProfile* user, const PersonalInfo& info) {
        std::cout << "\n=== Current Profile Information ===\n"
                 << "Name: " << user->getName() << "\n"
                 << "Date of Birth: " << info.getDateOfBirth() << "\n"
                 << "Blood Group: " << info.getBloodGroup() << "\n"
                 << "Phone: " << info.getPhone() << "\n"
                 << "Email: " << info.getEmail() << "\n";
    }

    static bool isValidPhoneNumber(const std::string& phone) {
        if (phone.length() != 10) return false;
        return std::all_of(phone.begin(), phone.end(), ::isdigit);
    }

    static bool isValidEmail(const std::string& email) {
        size_t atPos = email.find('@');
        if (atPos == std::string::npos) return false;
        size_t dotPos = email.find('.', atPos);
        if (dotPos == std::string::npos) return false;
        return atPos > 0 && dotPos > atPos + 1 && dotPos < email.length() - 1;
    }

    static bool isValidProvider(const std::string& provider) {
        return std::all_of(provider.begin(), provider.end(),
                          [](char c) { return std::isalpha(c) || std::isspace(c); });
    }

    static bool isValidPolicyNumber(const std::string& number) {
        return number.length() == 10 && std::all_of(number.begin(), number.end(), ::isdigit);
    }

    static int calculateAge(const std::string& dob) {
        int day = std::stoi(dob.substr(0, 2));
        int month = std::stoi(dob.substr(3, 2));
        int year = std::stoi(dob.substr(6, 4));

        time_t now = time(0);
        tm* ltm = localtime(&now);

        int currentYear = 1900 + ltm->tm_year;
        int currentMonth = 1 + ltm->tm_mon;
        int currentDay = ltm->tm_mday;

        int age = currentYear - year;
        if (currentMonth < month || (currentMonth == month && currentDay < day)) {
            age--;
        }
        return age;
    }
};




std::unordered_map<std::string, NurseProfile*> NurseProfile::nurseDatabase;
int NurseProfile::currentNurseID = 0;

void RecordManager::createDirectories(const std::string& patientId) {
    std::string basePath = "C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\PATH\\" + patientId;

    // Check if directory already exists
    if (std::filesystem::exists(basePath)) {
        Logger::log("Directory already exists: " + basePath);
        return;
    }

    // Create directory if it doesn't exist
    if (_mkdir(basePath.c_str()) == 0) {
        Logger::log("Successfully created directory: " + basePath);
    } else {
        Logger::log("Failed to create directory: " + basePath);
    }
}


bool RecordManager::fileExists(const std::string& filePath) {
    return std::filesystem::exists(filePath);
}

void RecordManager::copyBinaryFile(const std::string& sourcePath, const std::string& destinationDir) {
    // Create correct destination path including ID type prefix
    std::string baseDir = "C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\PATH";
    std::string fileName = std::filesystem::path(sourcePath).filename().string();
    std::string destPath = baseDir + "\\" + destinationDir + "\\" + fileName;

    try {
        std::filesystem::copy_file(sourcePath, destPath,
            std::filesystem::copy_options::overwrite_existing);
        fileRecords[destinationDir].push_back(fileName);
        Logger::log("Successfully copied file to: " + destPath);
        std::cout << "File copied successfully!\n";
    }
    catch (const std::filesystem::filesystem_error& e) {
        Logger::log("File copy error: " + std::string(e.what()));
        std::cout << "Error copying file: " << e.what() << "\n";
    }
}



void RecordManager::retrieveFile(const std::string& idType, const std::string& id) {
    std::string dirPath = "C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\PATH\\" +
                         idType + "_" + id;

    if (!std::filesystem::exists(dirPath)) {
        std::cout << "No records found for this ID\n";
        return;
    }

    std::cout << "Available files:\n";
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        std::cout << entry.path().filename().string() << "\n";
    }
}


void RecordManager::deleteFileFromQueue() {
    if (deleteQueue.empty()) {
        std::cout << "Delete queue is empty\n";
        return;
    }

    FileInfo fileInfo = deleteQueue.top();
    deleteQueue.pop();

    try {
        std::filesystem::remove(fileInfo.filePath);
        std::cout << "File deleted: " << fileInfo.filePath << "\n";
    } catch (const std::exception& e) {
        std::cout << "Error deleting file: " << e.what() << "\n";
    }
}

void RecordManager::listFiles(const std::string& idType, const std::string& id) {
    std::string dirPath = "C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\PATH\\" +
                         idType + "_" + id;
    if (!std::filesystem::exists(dirPath)) {
        std::cout << "Directory not found\n";
        return;
    }

    std::cout << "Files for " << idType << " " << id << ":\n";
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        auto timestamp = std::filesystem::last_write_time(entry.path());
        auto timeT = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now() +
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                timestamp.time_since_epoch()));
        std::cout << entry.path().filename().string() << " - Last modified: "
                 << std::ctime(&timeT);
    }
}


// After listFiles method
void RecordManager::addToDeleteQueue(const std::string& idType, const std::string& id, const std::string& fileName) {
    std::string filePath = "C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\PATH\\" +
                          idType + "_" + id + "\\" + fileName;

    if (!fileExists(filePath)) {
        std::cout << "File not found!\n";
        return;
    }

    auto timestamp = std::filesystem::last_write_time(filePath);
    auto timeT = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now() +
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            timestamp.time_since_epoch()));

    deleteQueue.push(FileInfo(filePath, timeT));
    std::cout << "File added to delete queue\n";
}

// Before showFileRecords method




void RecordManager::showFileRecords(const std::string& id) {
    auto it = fileRecords.find(id);
    if (it == fileRecords.end()) {
        std::cout << "No records found for this ID\n";
        return;
    }

    std::cout << "File records for ID " << id << ":\n";
    for (const auto& fileName : it->second) {
        std::cout << fileName << "\n";
    }
}


class Analytics {
private:
    static std::unordered_map<std::string, std::vector<MedicalRecord>> patientRecords;

public:
    static void displayUsers(bool isAuthorized = false) {
        std::ifstream file("C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\file.txt");
        std::string line;

        if (!isAuthorized) {
            while (getline(file, line)) {
                if (line.find("Name: ") != std::string::npos ||
                    line.find("Age: ") != std::string::npos) {
                    std::cout << line << "\n";
                }
            }
        } else {
            while (getline(file, line)) {
                std::cout << line << "\n";
            }
        }
        file.close();
    }

    static void generateHealthReport(const std::string& patientID) {
        auto records = patientRecords[patientID];
        double totalBill = 0;
        std::cout << "\n=== Health Report ===\n";
        for(const auto& record : records) {
            std::cout << "Date: " << record.getDate()
                      << "\nBill Amount: $" << record.getBillAmount() << "\n";
            totalBill += record.getBillAmount();
        }
        std::cout << "Total Medical Expenses: $" << totalBill << "\n";
    }

    static void searchPatients(const std::string& query) {
        for(const auto& [id, user] : UserProfile::getUserDatabase()) {
            if(user->getName().find(query) != std::string::npos) {
                std::cout << "ID: " << id << ", Name: " << user->getName() << "\n";
            }
        }
    }
};

// Initialize Analytics static member
std::unordered_map<std::string, std::vector<MedicalRecord>> Analytics::patientRecords;
void displayMenu() {
    std::cout << "\n=== Hospital Management System ===\n"
              << "1. Register New User\n"
              << "2. Register New Nurse\n"
              << "3. Search Patients\n"
              << "4. Update Personal Information\n"
              << "5. Display Users\n"
              << "6. Nurse Management\n"
              << "7. File Management\n"
              << "8. Exit\n\n"
              << "Choose an option: ";
}



void displayNurseMenu() {
    cout << "\n=== Nurse Management System ===\n"
         << "1. View Nurses\n"
         << "2. Rate Nurse\n"
         << "3. View Ratings\n"
         << "4. Top Rated Nurse\n"
         << "5. Search Nurse\n"
         << "6. Generate Report\n"
         << "7. Search Nurse by ID\n"    // New option
         << "8. Sort and Display Nurses\n" // New option
         << "9. Return to Main Menu\n"
         << "Choose an option: ";
}


// Add a new submenu for nurse management
void displayNurseMenu() {
    std::cout << "\n=== Nurse Management System ===\n"
              << "1. View Nurses\n"
              << "2. Rate Nurse\n"
              << "3. View Ratings\n"
              << "4. Top Rated Nurse\n"
              << "5. Search Nurse\n"
              << "6. Generate Report\n"
              << "7. Return to Main Menu\n"
              << "Choose an option: ";
}
void displayNurseMenu() {
    cout << "\n=== Nurse Management System ===\n"
         << "1. View Nurses\n"
         << "2. Rate Nurse\n"
         << "3. View Ratings\n"
         << "4. Top Rated Nurse\n"
         << "5. Search Nurse\n"
         << "6. Generate Report\n"
         << "7. Search Nurse by ID\n"    // New option
         << "8. Sort and Display Nurses\n" // New option
         << "9. Return to Main Menu\n"
         << "Choose an option: ";
}

// Replace the updatePersonalInfo function with this:
void updatePersonalInfo(const std::string& userID) {
    std::cout << "\nEnter password to verify: ";
    std::string password = UserInterface::getHiddenPassword();

    // Here you would verify the password against stored password
    // For demonstration, using a simple check
    if (password != "password123") {
        std::cout << "Invalid password. Access denied.\n";
        return;
    }

    std::string choice;
    do {
        std::cout << "\n=== Update Personal Information ===\n"
                  << "What would you like to update?\n\n"
                  << "1. Date of Birth\n"
                  << "2. Blood Group\n"
                  << "3. Phone Number\n"
                  << "4. Email Address\n"
                  << "5. Emergency Contact\n"
                  << "6. Address\n"
                  << "7. Go Back\n\n"
                  << "Choose an option: ";

        std::getline(std::cin, choice);

        UserProfile* user = UserProfile::getUser(userID);
        if (!user) {
            std::cout << "User not found!\n";
            return;
        }

        PersonalInfo currentInfo = user->getPersonalInfo();

        if (choice == "1") {
            std::string newDOB;
            std::cout << "Current Date of Birth: " << currentInfo.getDateOfBirth() << "\n";
            std::cout << "Enter new Date of Birth (DD/MM/YYYY): ";
            std::getline(std::cin, newDOB);
            if (UserInterface::isValidDate(newDOB)) {
                currentInfo.setDateOfBirth(newDOB);
                user->updatePersonalInfo(currentInfo);
                std::cout << "Date of Birth updated successfully!\n";
            }
        }
        // Add similar blocks for other options

    } while (choice != "7");
}


void loadUsersFromFile() {
    std::string filepath = "C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\file.txt";
    std::ifstream file(filepath);

    if (!file.is_open()) {
        return;
    }

    std::string line;
    std::string currentID;
    std::string currentName;
    const std::string defaultPassword = "Default@123";

    while (std::getline(file, line)) {
        if (line.find("=== USER PROFILE ===") != std::string::npos) {
            currentID = "";
            currentName = "";
        }
        else if (line.find("ID: ") != std::string::npos) {
            currentID = line.substr(4);
        }
        else if (line.find("Name: ") != std::string::npos && !currentID.empty()) {
            currentName = line.substr(6);
            try {
                UserProfile* user = new UserProfile(currentName, defaultPassword);
                const_cast<std::unordered_map<std::string, UserProfile*>&>(UserProfile::getUserDatabase())[currentID] = user;
            } catch (const std::exception& e) {
                // Silent exception handling
            }
        }
    }
    file.close();
}

// Binary Search implementation
int binarySearchNurse(const vector<NurseProfile*>& nurses, const string& targetId) {
    int left = 0;
    int right = nurses.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nurses[mid]->getNurseID() == targetId) {
            return mid;
        }
        if (nurses[mid]->getNurseID() < targetId) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

// Heapify function
void heapify(vector<NurseProfile*>& nurses, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && nurses[left]->getNurseID() > nurses[largest]->getNurseID()) {
        largest = left;
    }
    if (right < n && nurses[right]->getNurseID() > nurses[largest]->getNurseID()) {
        largest = right;
    }
    if (largest != i) {
        swap(nurses[i], nurses[largest]);
        heapify(nurses, n, largest);
    }
}

// Heap Sort implementation
void heapSort(vector<NurseProfile*>& nurses) {
    int n = nurses.size();

    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(nurses, n, i);
    }

    for (int i = n - 1; i > 0; i--) {
        swap(nurses[0], nurses[i]);
        heapify(nurses, i, 0);
    }
}



int main() {
    Logger::init();
    std::string choice;
    UserProfile::initialize();
    loadUsersFromFile();
    NurseManagement::loadNurseData();
    _mkdir("C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\PATH");

    system("cls && mode con: cols=100 lines=40");
    Sleep(1000);

    std::cout << "\n===============================================\n"
              << "  Welcome to the Medical Record Management System\n"
              << "===============================================\n\n";
    Sleep(200);

    std::cout << "Before you proceed, please review these important guidelines:\n\n";
    Sleep(200);

    std::cout << "USER PROFILE REQUIREMENTS:\n"
              << "1. Username: 3-20 characters\n"
              << "2. Password: Minimum 8 characters including:\n"
              << "   - Uppercase letters\n"
              << "   - Lowercase letters\n"
              << "   - Numbers\n"
              << "   - Special characters\n"
              << "3. Email: Valid format (example@domain.com)\n"
              << "4. Phone: 10 digits\n\n";
    Sleep(200);

    std::cout << "PATIENT PROFILE REQUIREMENTS:\n"
              << "1. Age: 0-120 years\n"
              << "2. Blood Group: A+, A-, B+, B-, O+, O-, AB+, AB-\n"
              << "3. Height: 50-250 cm\n"
              << "4. Weight: 1-500 kg\n\n";
    Sleep(200);

    std::cout << "FILE MANAGEMENT REQUIREMENTS:\n"
              << "1. Directory path: C:\\Users\\Asus\\OneDrive\\Desktop\\DAA_FINAL\\PATH\n"
              << "2. Supported file formats: pdf, jpg, png\n"
              << "3. File naming convention: [ID-type]_[ID-number]\n"
              << "4. Maximum file size: 10MB\n\n"
              << "Press Enter when you're ready to proceed...\n";

    std::cin.get();
    system("cls");

    std::cout << "\nHow are you? ";
    std::string response;
    std::getline(std::cin, response);

    system("cls");

    while(true) {
        displayMenu();
        std::getline(std::cin, choice);

        if(choice == "1") {
            NurseManagement::addNurse();
        }
        else if(choice == "2") {
            NurseManagement::viewNurses();
        }
        else if(choice == "3") {
            NurseManagement::rateNurse();
        }
        else if(choice == "4") {
            NurseManagement::viewRatings();
        }
        else if(choice == "5") {
            NurseManagement::searchNurse();
        }
        else if(choice == "6") {
            std::string nurseChoice;
            do {
                displayNurseMenu();
                std::getline(std::cin, nurseChoice);

                if(nurseChoice == "1") NurseManagement::viewNurses();
                else if(nurseChoice == "2") NurseManagement::rateNurse();
                else if(nurseChoice == "3") NurseManagement::viewRatings();
                else if(nurseChoice == "4") NurseManagement::topRatedNurse();
                else if(nurseChoice == "5") NurseManagement::searchNurse();
                else if(nurseChoice == "6") NurseManagement::generateDetailedReport();
            } while(nurseChoice != "7");
        }
        else if(choice == "7") {
            RecordManager manager;
            std::string fileChoice;
            do {
                displayFileMenu();
                std::getline(std::cin, fileChoice);

                if(fileChoice == "1") {
                    manager.handleFileStorage();
                }
                else if(fileChoice == "2") {
                    std::string idType, id;
                    std::cout << "Enter ID type (patient/nurse/caregiver): ";
                    std::getline(std::cin, idType);
                    std::cout << "Enter ID: ";
                    std::getline(std::cin, id);
                    manager.retrieveFile(idType, id);
                }
                else if(fileChoice == "3") {
                    std::string idType, id, fileName;
                    std::cout << "Enter ID type (patient/nurse/caregiver): ";
                    std::getline(std::cin, idType);
                    std::cout << "Enter ID: ";
                    std::getline(std::cin, id);
                    std::cout << "Enter filename to delete: ";
                    std::getline(std::cin, fileName);
                    manager.addToDeleteQueue(idType, id, fileName);
                    manager.deleteFileFromQueue();
                }
                else if(fileChoice == "4") {
                    std::string idType, id;
                    std::cout << "Enter ID type (patient/nurse/caregiver): ";
                    std::getline(std::cin, idType);
                    std::cout << "Enter ID: ";
                    std::getline(std::cin, id);
                    manager.listFiles(idType, id);
                }
                else if(fileChoice == "5") {
                    std::string id;
                    std::cout << "Enter ID: ";
                    std::getline(std::cin, id);
                    manager.showFileRecords(id);
                }
            } while(fileChoice != "6");
        }
        else if(choice == "8") {
            std::cout << "Thank you for using the system!\n";
            break;
        }
    }
    return 0;
}
