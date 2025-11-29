const mongoose = require('mongoose');

const bookingSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    amenity: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Amenity',
        required: true
    },
    date: {
        type: Date,
        required: [true, 'Booking date is required']
    },
    startTime: {
        type: String,
        required: [true, 'Start time is required']
    },
    endTime: {
        type: String,
        required: [true, 'End time is required']
    },
    status: {
        type: String,
        enum: ['confirmed', 'cancelled', 'completed'],
        default: 'confirmed'
    },
    notes: {
        type: String,
        default: ''
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

bookingSchema.index({ amenity: 1, date: 1, startTime: 1, endTime: 1 });

module.exports = mongoose.model('Booking', bookingSchema);
