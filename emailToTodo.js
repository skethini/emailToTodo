async function main() {    
    const text = document.getElementById("text");
    const convert = document.getElementById('convert');
    const result = document.getElementById("result");

    async function changeHTML() {
        console.log("Seeen");
        let inputValue = text.value; 
        console.log(inputValue);

        fetch(`http://localhost:5000/get_variable?input=${inputValue}`)
            .then(response => response.json())
            .then(data => {
                const variableValue = data.variable;
                console.log(`Received from Python: ${variableValue}`);

                const requests = variableValue[0];
                const questions = variableValue[1];

                result.innerHTML = "To-do List: <br>";
                for (let i = 0; i < requests.length; i++) {
                    const stringValue = requests[i];
                    result.innerHTML += stringValue + "<br>";
                }

                result.innerHTML += "<br>Questions List: <br>";
                for (let i = 0; i < questions.length; i++) {
                    const stringValue = questions[i];
                    result.innerHTML += stringValue + "<br>";
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }

    convert.addEventListener('click', async () => {
        changeHTML();
    });

    $(document).ready(function() {
        var inputElement = $('#text');
    
        function updateInputWidth() {
            var text = inputElement.val(); 
            var textWidth = inputElement[0].scrollWidth; 
            inputElement.css('width', textWidth + 'px');
        }
    
        inputElement.on('input', updateInputWidth);
    });
}

main();
