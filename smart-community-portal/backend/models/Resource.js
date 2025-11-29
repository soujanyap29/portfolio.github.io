const mongoose = require('mongoose');

const resourceSchema = new mongoose.Schema({
    owner: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    name: {
        type: String,
        required: [true, 'Resource name is required'],
        trim: true
    },
    description: {
        type: String,
        required: [true, 'Description is required']
    },
    category: {
        type: String,
        enum: ['tools', 'electronics', 'sports', 'books', 'household', 'other'],
        default: 'other'
    },
    imageUrl: {
        type: String,
        default: ''
    },
    availableFrom: {
        type: Date,
        default: Date.now
    },
    availableTo: {
        type: Date
    },
    status: {
        type: String,
        enum: ['available', 'borrowed', 'unavailable'],
        default: 'available'
    },
    currentBorrower: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        default: null
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Resource', resourceSchema);
