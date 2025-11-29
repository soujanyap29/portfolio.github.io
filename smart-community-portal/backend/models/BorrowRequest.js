const mongoose = require('mongoose');

const borrowRequestSchema = new mongoose.Schema({
    resource: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Resource',
        required: true
    },
    borrower: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    owner: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    message: {
        type: String,
        default: ''
    },
    status: {
        type: String,
        enum: ['pending', 'approved', 'rejected', 'returned'],
        default: 'pending'
    },
    borrowDate: {
        type: Date
    },
    returnDate: {
        type: Date
    },
    actualReturnDate: {
        type: Date
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('BorrowRequest', borrowRequestSchema);
