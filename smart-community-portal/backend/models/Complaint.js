const mongoose = require('mongoose');

const complaintSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    title: {
        type: String,
        required: [true, 'Complaint title is required'],
        trim: true
    },
    description: {
        type: String,
        required: [true, 'Complaint description is required']
    },
    category: {
        type: String,
        enum: ['maintenance', 'security', 'facilities', 'noise', 'parking', 'other'],
        default: 'other'
    },
    imageUrl: {
        type: String,
        default: ''
    },
    status: {
        type: String,
        enum: ['submitted', 'in_progress', 'resolved'],
        default: 'submitted'
    },
    priority: {
        type: String,
        enum: ['low', 'medium', 'high', 'urgent'],
        default: 'medium'
    },
    adminNotes: {
        type: String,
        default: ''
    },
    resolvedAt: {
        type: Date
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Complaint', complaintSchema);
