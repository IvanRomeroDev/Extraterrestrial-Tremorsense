document.getElementById('upload-form').addEventListener('sumbit', async function(event)
{
    event.preventDefault();
    
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];

    if(!file)
    {
        alert('Please select a CSV file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try
    {
        const response = await fetch('https://backend-url/process', {method: 'POST', body: formData});

        if(!response.ok)
        {
            throw new Error('Error Analyzing the Data.');
        }

        const result = await response.json();
        document.getElementById('result').innerText = JSON.stringify(result, null, 2);
    }
    catch(error)
    {
        console.error('Error:', error);
        document.getElementById('result').textContent = 'An error occured while processing the data.';
    }
});