const mongoose = require('mongoose');

const amenitySchema = new mongoose.Schema({
    name: {
        type: String,
        required: [true, 'Amenity name is required'],
        trim: true
    },
    description: {
        type: String,
        required: [true, 'Description is required']
    },
    location: {
        type: String,
        default: ''
    },
    operatingHours: {
        start: {
            type: String,
            default: '06:00'
        },
        end: {
            type: String,
            default: '22:00'
        }
    },
    capacity: {
        type: Number,
        default: 1
    },
    isActive: {
        type: Boolean,
        default: true
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Amenity', amenitySchema);
