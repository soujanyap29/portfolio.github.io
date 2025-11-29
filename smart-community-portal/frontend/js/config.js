const API_BASE_URL = 'http://localhost:4000/api';

const getToken = () => localStorage.getItem('token');

const getUser = () => {
    const userData = localStorage.getItem('user');
    return userData ? JSON.parse(userData) : null;
};

const isLoggedIn = () => !!getToken();

const isAdmin = () => {
    const user = getUser();
    return user && user.role === 'admin';
};
