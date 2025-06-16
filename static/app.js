document.addEventListener('DOMContentLoaded', function() {
    // Menu items data
    const menuItems = [
        { id: 1, name: '돼지국밥', price: 9000 },
        { id: 2, name: '순대국밥', price: 10000 },
        { id: 3, name: '내장국밥', price: 9500 },
        { id: 4, name: '섞어국밥', price: 9500 },
        { id: 5, name: '수육 반접시', price: 13000 },
        { id: 6, name: '수육 한접시', price: 25000 }
    ];

    let orderItems = [];
    const orderItemsContainer = document.querySelector('.order-items');
    const totalAmountElement = document.querySelector('.total-amount .amount');
    let recognition = null;
    let recognitionTimeout = null;  // 타이머를 전역 변수로 관리

    let currentSelectedItem = null;
    const modal = document.getElementById('quantityModal');
    const modalMenuName = document.getElementById('modalMenuName');
    const itemQuantity = document.getElementById('itemQuantity');
    const decreaseBtn = document.getElementById('decreaseQty');
    const increaseBtn = document.getElementById('increaseQty');
    const confirmBtn = document.getElementById('confirmAdd');
    const cancelBtn = document.getElementById('cancelAdd');

    // Initialize speech recognition
    function initSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.error('음성 인식을 지원하지 않는 브라우저입니다.');
            return null;
        }

        // 기존 인스턴스가 있으면 제거
        if (recognition) {
            recognition.stop();
            recognition = null;
        }

        recognition = new SpeechRecognition();
        recognition.continuous = false;  // 연속 인식 비활성화
        recognition.interimResults = false;  // 중간 결과 비활성화
        recognition.lang = 'ko-KR';

        recognition.onresult = async (event) => {
            const transcript = event.results[0][0].transcript;
            console.log('인식된 음성 (최종):', transcript);
            
            try {
                const response = await processVoiceCommand(transcript);
                
                if (response && response.order_items) {
                    Object.entries(response.order_items).forEach(([menuName, quantity]) => {
                        const menuItem = menuItems.find(item => menuName.includes(item.name));
                        if (menuItem) {
                            for (let i = 0; i < quantity; i++) {
                                addToOrder(menuItem);
                            }
                        }
                    });
                    speak(`주문이 추가되었습니다. ${transcript}`);
                }
            } catch (error) {
                console.error('음성 처리 오류:', error);
                speak('죄송합니다. 다시 말씀해주세요.');
            } finally {
                // 음성 인식 완료 후 자동으로 정리
                stopVoiceRecognition();
            }
        };

        recognition.onerror = (event) => {
            console.error('음성 인식 오류:', event.error);
            speak('음성 인식에 실패했습니다. 다시 시도해주세요.');
            stopVoiceRecognition();
        };

        recognition.onend = () => {
            // 음성 인식이 자연스럽게 종료될 때 정리
            if (recognition) {
                stopVoiceRecognition();
            }
        };

        return recognition;
    }

    // 음성으로 말하기
    function speak(text) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'ko-KR';
        window.speechSynthesis.speak(utterance);
    }

    // 서버에 음성 명령 전송
    async function processVoiceCommand(transcript) {
        try {
            const response = await fetch('/process-voice', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: transcript })
            });
            return await response.json();
        } catch (error) {
            console.error('음성 처리 요청 실패:', error);
            throw error;
        }
    }

    // 음성 인식 시작/중지
    function toggleVoiceMode() {
        if (!recognition) {
            recognition = initSpeechRecognition();
            if (!recognition) {
                alert('이 브라우저는 음성 인식을 지원하지 않습니다.');
                return;
            }
        }

        const voiceModeIndicator = document.querySelector('.voice-mode-indicator');
        const micIcon = document.querySelector('.mic-icon');
        
        if (recognition && recognition.state === 'listening') {
            stopVoiceRecognition();
        } else {
            startVoiceRecognition();
        }
    }

    // 음성 인식 시작
    function startVoiceRecognition() {
        const voiceModeIndicator = document.querySelector('.voice-mode-indicator');
        const micIcon = document.querySelector('.mic-icon');
        
        // 기존 타이머 초기화
        if (recognitionTimeout) {
            clearTimeout(recognitionTimeout);
            recognitionTimeout = null;
        }
        
        // 음성 인식 초기화
        recognition = initSpeechRecognition();
        if (!recognition) {
            alert('이 브라우저는 음성 인식을 지원하지 않습니다.');
            return;
        }
        
        try {
            recognition.start();
            voiceModeIndicator.style.display = 'flex';
            document.querySelector('.voice-control').style.display = 'flex';
            micIcon.textContent = 'mic_off';
            micIcon.style.color = 'red';
            speak('무엇을 주문하시겠어요?');

            // 5초 후 자동으로 음성 인식 종료
            recognitionTimeout = setTimeout(() => {
                if (recognition) {
                    speak('음성 인식이 종료되었습니다.');
                    stopVoiceRecognition();
                }
            }, 5000);
        } catch (error) {
            console.error('음성 인식 시작 오류:', error);
            speak('음성 인식을 시작할 수 없습니다.');
            stopVoiceRecognition();
        }
    }

    // 음성 인식 중지
    function stopVoiceRecognition() {
        const voiceModeIndicator = document.querySelector('.voice-mode-indicator');
        const micIcon = document.querySelector('.mic-icon');
        const voiceControl = document.querySelector('.voice-control');
        
        // 타이머 정리
        if (recognitionTimeout) {
            clearTimeout(recognitionTimeout);
            recognitionTimeout = null;
        }
        
        // 음성 인식 정지
        if (recognition) {
            try {
                recognition.stop();
            } catch (e) {
                console.error('음성 인식 정지 중 오류:', e);
            }
            recognition = null;
        }
        
        // UI 초기화
        voiceModeIndicator.style.display = 'none';
        voiceControl.style.display = 'none';
        if (micIcon) {
            micIcon.textContent = 'mic';
            micIcon.style.color = '';
        }
    }

    // Show modal with menu item
    function showQuantityModal(menuItem) {
        currentSelectedItem = menuItem;
        modalMenuName.textContent = `${menuItem.name} (${menuItem.price.toLocaleString()}원)`;
        itemQuantity.value = 1;
        modal.style.display = 'flex';
    }

    // Hide modal
    function hideQuantityModal() {
        modal.style.display = 'none';
        currentSelectedItem = null;
    }

    // Quantity controls
    decreaseBtn.addEventListener('click', () => {
        const value = parseInt(itemQuantity.value);
        if (value > 1) {
            itemQuantity.value = value - 1;
        }
    });

    increaseBtn.addEventListener('click', () => {
        const value = parseInt(itemQuantity.value);
        if (value < 99) {
            itemQuantity.value = value + 1;
        }
    });

    // Confirm button click
    confirmBtn.addEventListener('click', () => {
        if (currentSelectedItem) {
            const quantity = parseInt(itemQuantity.value);
            for (let i = 0; i < quantity; i++) {
                addToOrder(currentSelectedItem);
            }
            hideQuantityModal();
            speak(`${currentSelectedItem.name} ${quantity}개를 주문에 추가했습니다.`);
        }
    });

    // Cancel button click
    cancelBtn.addEventListener('click', hideQuantityModal);

    // Close modal when clicking outside
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            hideQuantityModal();
        }
    });

    // Add item to order
    function addToOrder(menuItem) {
        const existingItem = orderItems.find(item => item.id === menuItem.id);
        
        if (existingItem) {
            existingItem.quantity += 1;
        } else {
            orderItems.push({
                ...menuItem,
                quantity: 1
            });
        }
        
        updateOrderDisplay();
    }

    // Update the order display
    function updateOrderDisplay() {
        orderItemsContainer.innerHTML = '';
        let total = 0;
        
        orderItems.forEach(item => {
            const itemTotal = item.price * item.quantity;
            total += itemTotal;
            
            const itemElement = document.createElement('div');
            itemElement.className = 'order-item';
            itemElement.innerHTML = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span>${item.name} x${item.quantity}</span>
                    <span>${itemTotal.toLocaleString()}원</span>
                </div>
            `;
            orderItemsContainer.appendChild(itemElement);
        });
        
        totalAmountElement.textContent = `${total.toLocaleString()}원`;
    }

    // Update menu item click handlers
    document.querySelectorAll('.menu-item').forEach((item, index) => {
        item.addEventListener('click', () => {
            showQuantityModal(menuItems[index]);
        });
    });

    // Voice order button click handler
    document.querySelector('.voice-order').addEventListener('click', toggleVoiceMode);
    document.querySelector('.voice-control').addEventListener('click', toggleVoiceMode);

    // Complete order button click handler
    document.querySelector('.complete-order').addEventListener('click', function() {
        if (orderItems.length === 0) {
            alert('주문할 메뉴를 선택해주세요.');
            return;
        }
        
        const orderSummary = orderItems.map(item => 
            `${item.name} ${item.quantity}개`
        ).join(', ');
        
        if (confirm(`주문하시겠습니까?\n\n${orderSummary}\n\n총 금액: ${orderItems.reduce((sum, item) => sum + (item.price * item.quantity), 0).toLocaleString()}원`)) {
            alert('주문이 완료되었습니다!');
            orderItems = [];
            updateOrderDisplay();
        }
    });
});
