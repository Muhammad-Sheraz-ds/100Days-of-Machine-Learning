## Project Title: Library Management System

### Project Description:

The Library Management System allows users (teachers, admin, and students) to interact with the library's catalog, manage books, and perform various tasks related to borrowing and returning books. The project will use file handling to store data and dictionaries to manage book records and user information.

### Project Modules:

**User Authentication:**
- Implement a login system for teachers, admin, and students using username and password authentication. Use file handling to store and validate user credentials.

**Book Management:**
- Create a module for adding new books to the library, updating book information, deleting books, and searching for books based on titles, authors, or categories. Store book data in a file (e.g., CSV or JSON) using file handling concepts.

**User Modules:**
- **Admin Module:** Allow the admin to view and manage all user accounts. Admins can add new users, update user information, and deactivate user accounts.
- **Teacher Module:** Teachers can view the catalog, borrow books, and return books. They can also view their borrowing history.
- **Student Module:** Students can search for books, borrow books (limited to a specific number), return books, and view their borrowing history.

**Data Storage:**
- Use dictionaries to store book information, user details, and borrowing history. Serialize dictionaries and store them in files for persistent data storage.

**Error Handling:**
- Implement error handling for various scenarios, such as incorrect login credentials, attempting to borrow more books than allowed, or searching for a non-existing book.

**User-Friendly Interface:**
- Create a menu-driven interface for users to interact with the system. Use functions to modularize the code and enhance readability.

**Assessment Criteria:**
- Functionality
- Code Quality
- Data Management
- User Experience