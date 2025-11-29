require('dotenv').config();
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const User = require('./models/User');
const Amenity = require('./models/Amenity');
const Resource = require('./models/Resource');
const Announcement = require('./models/Announcement');
const Complaint = require('./models/Complaint');

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/smart_community';

const seedData = async () => {
    try {
        await mongoose.connect(MONGO_URI);
        console.log('Connected to MongoDB');

        await User.deleteMany({});
        await Amenity.deleteMany({});
        await Resource.deleteMany({});
        await Announcement.deleteMany({});
        await Complaint.deleteMany({});

        console.log('Cleared existing data');

        const admin = await User.create({
            name: 'Admin User',
            email: 'admin@community.com',
            phone: '9876543210',
            flatNumber: 'A-101',
            password: 'Admin@123',
            role: 'admin'
        });
        console.log('Admin user created:', admin.email);

        const residents = await User.create([
            {
                name: 'John Doe',
                email: 'john@resident.com',
                phone: '9876543211',
                flatNumber: 'B-201',
                password: 'Password@123',
                role: 'resident'
            },
            {
                name: 'Jane Smith',
                email: 'jane@resident.com',
                phone: '9876543212',
                flatNumber: 'C-301',
                password: 'Password@123',
                role: 'resident'
            },
            {
                name: 'Mike Wilson',
                email: 'mike@resident.com',
                phone: '9876543213',
                flatNumber: 'D-401',
                password: 'Password@123',
                role: 'resident'
            }
        ]);
        console.log(`${residents.length} resident users created`);

        const amenities = await Amenity.create([
            {
                name: 'Gym',
                description: 'Fully equipped fitness center with modern equipment',
                location: 'Ground Floor, Block A',
                operatingHours: { start: '06:00', end: '22:00' },
                capacity: 20
            },
            {
                name: 'Clubhouse',
                description: 'Community hall for events and gatherings',
                location: 'First Floor, Main Building',
                operatingHours: { start: '09:00', end: '21:00' },
                capacity: 100
            },
            {
                name: 'Swimming Pool',
                description: 'Olympic-size swimming pool with changing rooms',
                location: 'Behind Block B',
                operatingHours: { start: '06:00', end: '20:00' },
                capacity: 30
            },
            {
                name: 'Tennis Court',
                description: 'Professional tennis court with lighting',
                location: 'Sports Complex',
                operatingHours: { start: '06:00', end: '21:00' },
                capacity: 4
            },
            {
                name: 'Guest Room',
                description: 'Comfortable guest accommodation for visitors',
                location: 'Second Floor, Block A',
                operatingHours: { start: '00:00', end: '23:59' },
                capacity: 2
            }
        ]);
        console.log(`${amenities.length} amenities created`);

        const resources = await Resource.create([
            {
                owner: residents[0]._id,
                name: 'Power Drill',
                description: 'Cordless power drill with various bits',
                category: 'tools',
                status: 'available'
            },
            {
                owner: residents[1]._id,
                name: 'Board Games Collection',
                description: 'Collection including Monopoly, Scrabble, and Chess',
                category: 'other',
                status: 'available'
            },
            {
                owner: residents[0]._id,
                name: 'Camping Tent',
                description: '4-person camping tent, waterproof',
                category: 'sports',
                status: 'available'
            },
            {
                owner: residents[2]._id,
                name: 'Pressure Cooker',
                description: '5L stainless steel pressure cooker',
                category: 'household',
                status: 'available'
            }
        ]);
        console.log(`${resources.length} resources created`);

        const announcements = await Announcement.create([
            {
                title: 'Water Supply Maintenance',
                content: 'Water supply will be interrupted on Sunday from 10 AM to 2 PM for routine maintenance work.',
                category: 'maintenance',
                priority: 'high',
                author: admin._id
            },
            {
                title: 'Community Diwali Celebration',
                content: 'Join us for the annual Diwali celebration at the clubhouse on November 12th at 6 PM. All residents are welcome!',
                category: 'event',
                priority: 'medium',
                author: admin._id
            },
            {
                title: 'New Parking Rules',
                content: 'Please note the updated parking guidelines. All vehicles must be parked in designated spots only.',
                category: 'policy',
                priority: 'medium',
                author: admin._id
            }
        ]);
        console.log(`${announcements.length} announcements created`);

        const complaints = await Complaint.create([
            {
                user: residents[0]._id,
                title: 'Broken Street Light',
                description: 'The street light near Block B entrance has been non-functional for a week.',
                category: 'maintenance',
                priority: 'medium',
                status: 'submitted'
            },
            {
                user: residents[1]._id,
                title: 'Noise from Construction',
                description: 'Construction work is happening beyond permitted hours causing disturbance.',
                category: 'noise',
                priority: 'high',
                status: 'in_progress',
                adminNotes: 'Contacted construction team to follow timings'
            }
        ]);
        console.log(`${complaints.length} complaints created`);

        console.log('\n=== Seed Data Summary ===');
        console.log('Admin Login: admin@community.com / Admin@123');
        console.log('Resident Logins:');
        console.log('  - john@resident.com / Password@123');
        console.log('  - jane@resident.com / Password@123');
        console.log('  - mike@resident.com / Password@123');
        console.log('=========================\n');

        console.log('Database seeding completed successfully!');
        process.exit(0);
    } catch (error) {
        console.error('Error seeding database:', error);
        process.exit(1);
    }
};

seedData();
