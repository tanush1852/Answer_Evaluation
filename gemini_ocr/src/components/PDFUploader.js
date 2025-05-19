import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemText,
  Grid,
  Chip,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import axios from 'axios';

const PDFUploader = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [formattedData, setFormattedData] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    const uploadedFile = acceptedFiles[0];
    if (uploadedFile && uploadedFile.type === 'application/pdf') {
      setFile(uploadedFile);
      setError(null);
      setFormattedData(null);
      setLoading(true);

      const formData = new FormData();
      formData.append('pdf', uploadedFile);

      try {
        const response = await axios.post('http://localhost:5000/extract-text', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        setFormattedData(response.data.formatted_data);
        console.log('Extracted Data:', JSON.stringify(response.data.extracted_data, null, 2));
      } catch (err) {
        setError(err.response?.data?.message || 'Error processing PDF');
      } finally {
        setLoading(false);
      }
    } else {
      setError('Please upload a valid PDF file');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    multiple: false,
  });

  const renderQuestion = (item, index) => {
    return (
      <Box key={index} sx={{ mb: 3 }}>
        <Typography variant="h6" color="primary" gutterBottom>
          {item.question || 'Additional Content'}
        </Typography>
        <List>
          {item.subquestions.map((subquestion, subIndex) => (
            <ListItem key={subIndex} sx={{ pl: 4 }}>
              <ListItemText
                primary={
                  <Box>
                    <Typography variant="subtitle1" component="span" color="text.secondary">
                      {subquestion.label ? `${subquestion.label}.` : ''}
                    </Typography>
                    <Typography variant="body1" component="div" sx={{ mt: 1 }}>
                      {subquestion.question}
                    </Typography>
                  </Box>
                }
                secondary={
                  <Typography 
                    variant="body1" 
                    component="div" 
                    sx={{ 
                      whiteSpace: 'pre-line',
                      mt: 1,
                      color: 'text.primary',
                      fontWeight: 'medium'
                    }}
                  >
                    {subquestion.content || 'No answer provided yet'}
                  </Typography>
                }
              />
            </ListItem>
          ))}
        </List>
        <Divider sx={{ mt: 2 }} />
      </Box>
    );
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'divider',
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          {isDragActive
            ? 'Drop the PDF here'
            : 'Drag and drop a PDF file here, or click to select'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Only PDF files are accepted
        </Typography>
      </Paper>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {formattedData && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Extracted Content with Comparison
          </Typography>
          <Card>
            <CardContent>
              {formattedData.map((item) => renderQuestion(item))}
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
};

export default PDFUploader;